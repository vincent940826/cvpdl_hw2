"""TTA methods. Add your own here and register it in METHODS.

A method is a callable: method(x) -> LongTensor of labels in {0..19} for the normalized batch x.
It may update its own state (BN statistics, parameters, memory, ...) as the stream goes on.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Source:
    """No adaptation."""

    def __init__(self, model):
        self.model = model.eval()

    @torch.no_grad()
    def __call__(self, x):
        self.last_logits = self.model(x)
        return self.last_logits.argmax(1)


def use_batch_stats(model):
    """Make every BN layer normalize with the current batch's mean/var instead of the stored ones."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None


class BNAdapt:
    """Test BN: BN statistics from the current batch, no parameter update (Schneider et al., 2020)."""

    def __init__(self, model):
        self.model = model.eval()
        use_batch_stats(self.model)

    @torch.no_grad()
    def __call__(self, x):
        self.last_logits = self.model(x)
        return self.last_logits.argmax(1)


class Tent:
    """Test BN + one entropy-minimization step on the BN affine params per batch (Wang et al., ICLR 2021).

    Optimizer follows the paper's ImageNet setting (SGD, lr 0.00025, momentum 0.9, batch 64).
    The prediction comes from the forward pass before the update, as in the official code.
    """

    def __init__(self, model, lr=0.00025, momentum=0.9):
        self.model = model.eval()
        use_batch_stats(self.model)
        self.model.requires_grad_(False)
        params = []
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                params += [m.weight, m.bias]
        self.optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)

    def __call__(self, x):
        logits = self.model(x)
        entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()
        entropy.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.last_logits = logits.detach()
        return self.last_logits.argmax(1)


class BlendBN(nn.Module):
    """BN that normalizes with (1 - alpha) * reference stats + alpha * current-batch stats.

    The reference starts as the source (stored) stats. With update=True the blended stats become the new
    reference after every batch, i.e. an exponential moving average over the stream with momentum alpha.
    alpha=0 is `source`, alpha=1 is `bn_adapt`.
    """

    def __init__(self, bn, alpha, update):
        super().__init__()
        self.bn, self.alpha, self.update = bn, alpha, update
        self.register_buffer("ref_mean", bn.running_mean.detach().clone())
        self.register_buffer("ref_var", bn.running_var.detach().clone())

    def forward(self, x):
        a = self.alpha
        bmean, bvar = x.mean((0, 2, 3)), x.var((0, 2, 3), unbiased=False)
        # Channel-averaged symmetric KL between batch and reference stats (read by BNAuto)
        d2, v1, v2 = (bmean - self.ref_mean) ** 2, bvar + 1e-5, self.ref_var + 1e-5
        self.shift = (0.5 * ((v1 + d2) / v2 + (v2 + d2) / v1) - 1).mean()
        mean = (1 - a) * self.ref_mean + a * bmean
        var = (1 - a) * self.ref_var + a * bvar
        if self.update:
            self.ref_mean, self.ref_var = mean.detach(), var.detach()
        return F.batch_norm(x, mean, var, self.bn.weight, self.bn.bias, False, 0.0, self.bn.eps)


def replace_bn(model, make):
    """Swap every BatchNorm2d in `model` for make(bn)."""
    for name, m in list(model.named_modules()):
        if isinstance(m, nn.BatchNorm2d):
            parent, _, attr = name.rpartition(".")
            setattr(model.get_submodule(parent), attr, make(m))


def prior_correction(logits):
    """ROID's output-side label-prior correction (Marsden et al., WACV 2024). Returns log-probs.

    Reweights each prediction by a smoothed estimate of the batch's class proportions.
    """
    p = logits.softmax(1)
    n, c = p.shape
    prior = p.mean(0)
    smooth = max(1 / n, 1 / c) / prior.max()
    prior = (prior + smooth) / (1 + smooth * c)
    q = p * prior
    return (q / q.sum(1, keepdim=True)).log()


def lame(logits, feats, k=5, steps=100):
    """LAME (Boudiaf et al., CVPR 2022): Laplacian-regularized assignment on a kNN graph. Returns log-probs.

    Images whose features are close are pushed toward the same class; the model itself is not changed.
    """
    f = F.normalize(feats.float(), dim=1)
    knn = (f @ f.T).topk(k + 1, dim=1).indices[:, 1:]  # drop self
    w = torch.zeros(len(f), len(f), device=f.device).scatter_(1, knn, 1.0)
    unary = logits.float().log_softmax(1)
    z = unary.softmax(1)
    for _ in range(steps):
        z = (unary + w @ z).softmax(1)
    return z.clamp_min(1e-12).log()


class BNMix:
    """Fixed blend of source and current-batch BN stats (Schneider et al., 2020, prior strength).

    post: "none", "prior" (ROID prior correction) or "lame" (LAME on penultimate features, kNN size k).
    """

    def __init__(self, model, alpha=0.3, post="none", k=5):
        self.model = model.eval()
        replace_bn(self.model, lambda bn: BlendBN(bn, alpha, update=False))
        self.post, self.k = post, k
        if post == "lame":
            self.model.fc.register_forward_hook(lambda m, inp, out: setattr(self, "feats", inp[0]))

    @torch.no_grad()
    def __call__(self, x):
        logits = self.model(x)
        if self.post == "prior":
            logits = prior_correction(logits)
        elif self.post == "lame":
            logits = lame(logits, self.feats, self.k)
        self.last_logits = logits
        return logits.argmax(1)


class BNEma:
    """BN stats as an EMA over the stream, starting from the source stats (in the spirit of RoTTA's RBN)."""

    def __init__(self, model, momentum=0.1):
        self.model = model.eval()
        replace_bn(self.model, lambda bn: BlendBN(bn, momentum, update=True))

    @torch.no_grad()
    def __call__(self, x):
        self.last_logits = self.model(x)
        return self.last_logits.argmax(1)


class BNAuto:
    """bn_mix with a per-batch alpha set from how far the batch is from the source domain.

    Pass 1 (alpha = 0, i.e. source stats) measures d = mean shift of the last k BN layers' input stats
    from the source stats; deep layers track how much the source model is disturbed. Pass 2 predicts with
    alpha = a_max * min(1, d / d0): mild domains get a small alpha, anything beyond d0 gets a_max.
    """

    def __init__(self, model, a_max=0.3, d0=1.0, k=5):
        self.model = model.eval()
        replace_bn(self.model, lambda bn: BlendBN(bn, 0.0, update=False))
        self.bns = [m for m in self.model.modules() if isinstance(m, BlendBN)]
        self.a_max, self.d0, self.k = a_max, d0, k
        self.alphas = []

    def set_alpha(self, a):
        for m in self.bns:
            m.alpha = a

    @torch.no_grad()
    def __call__(self, x):
        self.set_alpha(0.0)
        self.model(x)
        d = torch.stack([m.shift for m in self.bns[-self.k:]]).mean().item()
        a = self.a_max * min(1.0, d / self.d0)
        self.alphas.append(a)
        self.set_alpha(a)
        self.last_logits = self.model(x)
        return self.last_logits.argmax(1)


METHODS = {"source": Source, "bn_adapt": BNAdapt, "tent": Tent, "bn_mix": BNMix, "bn_ema": BNEma, "bn_auto": BNAuto}
