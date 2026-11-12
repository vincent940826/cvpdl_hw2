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
        mean = (1 - a) * self.ref_mean + a * x.mean((0, 2, 3))
        var = (1 - a) * self.ref_var + a * x.var((0, 2, 3), unbiased=False)
        if self.update:
            self.ref_mean, self.ref_var = mean.detach(), var.detach()
        return F.batch_norm(x, mean, var, self.bn.weight, self.bn.bias, False, 0.0, self.bn.eps)


def replace_bn(model, make):
    """Swap every BatchNorm2d in `model` for make(bn)."""
    for name, m in list(model.named_modules()):
        if isinstance(m, nn.BatchNorm2d):
            parent, _, attr = name.rpartition(".")
            setattr(model.get_submodule(parent), attr, make(m))


class BNMix:
    """Fixed blend of source and current-batch BN stats (Schneider et al., 2020, prior strength)."""

    def __init__(self, model, alpha=0.3):
        self.model = model.eval()
        replace_bn(self.model, lambda bn: BlendBN(bn, alpha, update=False))

    @torch.no_grad()
    def __call__(self, x):
        self.last_logits = self.model(x)
        return self.last_logits.argmax(1)


class BNEma:
    """BN stats as an EMA over the stream, starting from the source stats (in the spirit of RoTTA's RBN)."""

    def __init__(self, model, momentum=0.1):
        self.model = model.eval()
        replace_bn(self.model, lambda bn: BlendBN(bn, momentum, update=True))

    @torch.no_grad()
    def __call__(self, x):
        self.last_logits = self.model(x)
        return self.last_logits.argmax(1)


METHODS = {"source": Source, "bn_adapt": BNAdapt, "tent": Tent, "bn_mix": BNMix, "bn_ema": BNEma}
