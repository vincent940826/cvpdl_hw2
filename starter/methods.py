"""TTA methods. Add your own here and register it in METHODS.

A method is a callable: method(x) -> LongTensor of labels in {0..19} for the normalized batch x.
It may update its own state (BN statistics, parameters, memory, ...) as the stream goes on.
"""
import copy
import math
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # same normalization as harness.py
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


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
        self.weights = None  # optional per-sample weights for the batch stats (set by BNMix(balance=True))

    def forward(self, x):
        a = self.alpha
        if self.weights is None:
            bmean, bvar = x.mean((0, 2, 3)), x.var((0, 2, 3), unbiased=False)
        else:
            w = self.weights.view(-1, 1)
            bmean = (x.mean((2, 3)) * w).sum(0)
            bvar = (((x - bmean.view(1, -1, 1, 1)) ** 2).mean((2, 3)) * w).sum(0)
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
    balance=True (our addition): predict once, then recompute the batch stats with each image weighted by
    1 / (number of images in the batch with its predicted class), so every predicted class counts equally, and
    predict again. Targets the label bias of non-i.i.d. batch stats without using past batches.
    """

    def __init__(self, model, alpha=0.3, post="none", k=5, balance=False):
        self.model = model.eval()
        replace_bn(self.model, lambda bn: BlendBN(bn, alpha, update=False))
        self.bns = [m for m in self.model.modules() if isinstance(m, BlendBN)]
        self.post, self.k, self.balance = post, k, balance
        if post == "lame":
            self.model.fc.register_forward_hook(lambda m, inp, out: setattr(self, "feats", inp[0]))

    def predict(self, x):
        logits = self.model(x)
        if self.post == "prior":
            logits = prior_correction(logits)
        elif self.post == "lame":
            logits = lame(logits, self.feats, self.k)
        return logits

    @torch.no_grad()
    def __call__(self, x):
        logits = self.predict(x)
        if self.balance:
            cls = logits.argmax(1)
            w = 1 / torch.bincount(cls, minlength=logits.shape[1]).float()[cls]
            for m in self.bns:
                m.weights = w / w.sum()
            logits = self.predict(x)
            for m in self.bns:
                m.weights = None
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


class RobustBN(nn.Module):
    """RoTTA's robust BN (RBN): normalize with global stats; in train mode, first EMA-update them with the input's
    stats (the input is the memory bank, not the test batch)."""

    def __init__(self, bn, momentum, anchor=1.0):
        super().__init__()
        self.bn, self.momentum, self.anchor = bn, momentum, anchor
        self.register_buffer("mean", bn.running_mean.detach().clone())
        self.register_buffer("var", bn.running_var.detach().clone())
        self.register_buffer("src_mean", bn.running_mean.detach().clone())
        self.register_buffer("src_var", bn.running_var.detach().clone())
        self.watch, self.prev, self.dprev = False, None, None  # domain-shift watch (RoTTA reset_k, our addition)

    def forward(self, x):
        if not self.training:
            if self.watch:  # symmetric KL between this batch's input stats and the previous batch's
                bm, bv = x.mean((0, 2, 3)), x.var((0, 2, 3), unbiased=False) + 1e-5
                if self.prev is not None:
                    pm, pv = self.prev
                    d2 = (bm - pm) ** 2
                    self.dprev = (0.5 * ((bv + d2) / pv + (pv + d2) / bv) - 1).mean()
                self.prev = (bm, bv)
            mean, var, b = self.mean, self.var, self.anchor  # anchor < 1: our addition, blend back toward source
            if b < 1:
                mean, var = (1 - b) * self.src_mean + b * mean, (1 - b) * self.src_var + b * var
            return F.batch_norm(x, mean, var, self.bn.weight, self.bn.bias, False, 0.0, self.bn.eps)
        m = self.momentum
        mean = (1 - m) * self.mean + m * x.mean((0, 2, 3))
        var = (1 - m) * self.var + m * x.var((0, 2, 3), unbiased=False)
        self.mean, self.var = mean.detach(), var.detach()
        # Normalize by hand: cuDNN's batch_norm rejects stats that carry gradients (as in the official RBN)
        x = (x - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + self.bn.eps)
        return x * self.bn.weight.view(1, -1, 1, 1) + self.bn.bias.view(1, -1, 1, 1)


class CSTU:
    """RoTTA's category-balanced memory: slots are split across predicted classes; when full, the oldest and most
    uncertain sample (score = lambda_t * sigmoid(age / capacity) + lambda_u * entropy / log C) is evicted first."""

    def __init__(self, capacity=64, num_class=20, lambda_t=1.0, lambda_u=1.0):
        self.cap, self.C, self.per_class = capacity, num_class, capacity / num_class
        self.lt, self.lu = lambda_t, lambda_u
        self.data = [[] for _ in range(num_class)]  # items: [image, entropy, birth time]
        self.t = 0  # one tick per added sample, so age = t - birth

    def score(self, item):
        return self.lt / (1 + math.exp(-(self.t - item[2]) / self.cap)) + self.lu * item[1] / math.log(self.C)

    def add(self, x, cls, ent):
        new = [x, ent, self.t]
        if self.make_room(cls, self.score(new)):
            self.data[cls].append(new)
        self.t += 1

    def make_room(self, cls, new_score):
        occ = [len(items) for items in self.data]
        if occ[cls] < self.per_class:
            if sum(occ) < self.cap:
                return True
            classes = [c for c in range(self.C) if occ[c] == max(occ)]  # evict from the majority classes
        else:
            classes = [cls]
        worst = None
        for c in classes:
            for i, item in enumerate(self.data[c]):
                s = self.score(item)
                if worst is None or s >= worst[0]:
                    worst = (s, c, i)
        if worst is None:
            return True
        if worst[0] > new_score:
            self.data[worst[1]].pop(worst[2])
            return True
        return False

    def get(self):
        items = [item for c in self.data for item in c]
        x = torch.stack([item[0] for item in items])
        age = torch.tensor([(self.t - item[2]) / self.cap for item in items], device=x.device)
        return x, age


def strong_aug(x):
    """CoTTA-style strong augmentation on a normalized GPU batch: random affine (rotation +-15 deg, translate +-1/16,
    scale 0.9-1.1) with horizontal flip, brightness / contrast / saturation jitter, small Gaussian noise."""
    n, dev = len(x), x.device
    mean, std = MEAN.to(dev), STD.to(dev)
    img = x * std + mean
    u = lambda lo, hi: torch.rand(n, device=dev) * (hi - lo) + lo  # noqa: E731
    ang, sc = u(-15, 15) * math.pi / 180, u(0.9, 1.1)
    flip = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos * flip, -sin, u(-1 / 8, 1 / 8)], 1),
                         torch.stack([sin * flip, cos, u(-1 / 8, 1 / 8)], 1)], 1)
    img = F.grid_sample(img, F.affine_grid(theta, list(img.shape), align_corners=False),
                        padding_mode="reflection", align_corners=False)
    lum = torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)
    img = img * u(0.6, 1.4).view(n, 1, 1, 1)
    m = (img * lum).sum(1, keepdim=True).mean((2, 3), keepdim=True)
    img = (img - m) * u(0.7, 1.3).view(n, 1, 1, 1) + m
    g = (img * lum).sum(1, keepdim=True)
    img = (img - g) * u(0.5, 1.5).view(n, 1, 1, 1) + g
    img = (img + 0.005 * torch.randn_like(img)).clamp(0, 1)
    return (img - mean) / std


class RoTTA:
    """RoTTA (Yuan et al., CVPR 2023): RBN + category-balanced memory (CSTU) + teacher-student training on the memory.

    Defaults follow the paper: alpha (RBN momentum) 0.05, nu (teacher EMA) 0.001, Adam lr 1e-3, memory 64,
    lambda_t = lambda_u = 1, BN affine params trained, timeliness-weighted loss, prediction from the teacher.
    Our switches: train=False keeps only the statistics part (no gradients); update_first=True adds the current batch
    to the memory and updates before predicting it (the paper predicts first); post="lame" applies LAME to the output;
    anchor < 1 predicts with (1 - anchor) * source stats + anchor * RBN stats.
    reset_k (our addition): when the shallow-layer stats jump vs the previous batch by more than reset_k x the median
    jump seen so far, empty the memory and let the next RBN update replace the stats fully (momentum 1).
    """

    def __init__(self, model, alpha=0.05, nu=0.001, lr=1e-3, capacity=64, lambda_t=1.0, lambda_u=1.0,
                 train=True, update_first=False, post="none", k=5, anchor=1.0, reset_k=None):
        replace_bn(model, lambda bn: RobustBN(bn, alpha, anchor))
        model.requires_grad_(False)
        self.teacher = copy.deepcopy(model).eval()
        self.model = self.teacher  # what run.py --diag inspects
        self.student = None
        if train:
            self.student = model
            params = []
            for m in self.student.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
                    params += [m.weight, m.bias]
            self.opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
        self.mem = CSTU(capacity, 20, lambda_t, lambda_u)
        self.nu, self.update_first, self.post, self.k = nu, update_first, post, k
        self.rbns = [m for net in (self.teacher, self.student) if net is not None
                     for m in net.modules() if isinstance(m, RobustBN)]
        t_rbns = [m for m in self.teacher.modules() if isinstance(m, RobustBN)]
        self.watched = t_rbns[:len(t_rbns) // 3]  # shallow layers: corruption shows here, labels barely do
        self.reset_k, self.hist, self.resets, self.nbatch, self.full_update = reset_k, [], [], 0, False
        if post == "lame":
            self.teacher.fc.register_forward_hook(lambda m, inp, out: setattr(self, "feats", inp[0]))

    def finish(self, logits):
        return lame(logits, self.feats, self.k) if self.post == "lame" else logits

    def remember(self, x, logits):
        p = logits.softmax(1)
        ent = -(p * (p + 1e-6).log()).sum(1)
        for xi, c, e in zip(x, p.argmax(1).tolist(), ent.tolist()):
            self.mem.add(xi.clone(), c, e)

    def check_shift(self):
        ds = [m.dprev for m in self.watched if m.dprev is not None]
        if ds:
            d = torch.stack(ds).mean().item()
            if len(self.hist) >= 3 and d > self.reset_k * statistics.median(self.hist):
                self.mem.data = [[] for _ in range(self.mem.C)]
                self.full_update = True
                self.resets.append(self.nbatch)
            self.hist.append(d)
        self.nbatch += 1

    def update(self):
        xm, age = self.mem.get()
        if self.full_update:
            for m in self.rbns:
                m.momentum_saved, m.momentum = m.momentum, 1.0
        self.teacher.train()
        with torch.no_grad():
            t_out = self.teacher(xm)  # also EMA-updates the teacher's RBN stats with the memory's stats
        self.teacher.eval()
        if self.full_update and self.student is None:
            for m in self.rbns:
                m.momentum = m.momentum_saved
            self.full_update = False
        if self.student is None:
            return
        self.student.train()
        s_out = self.student(strong_aug(xm))
        w = torch.exp(-age) / (1 + torch.exp(-age))  # timeliness: newer samples weigh more
        loss = (-(t_out.softmax(1) * s_out.log_softmax(1)).sum(1) * w).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            for tp, sp in zip(self.teacher.parameters(), self.student.parameters()):
                tp.mul_(1 - self.nu).add_(sp.detach(), alpha=self.nu)
        if self.full_update:
            for m in self.rbns:
                m.momentum = m.momentum_saved
            self.full_update = False

    def __call__(self, x):
        for m in self.watched:
            m.watch = self.reset_k is not None
        with torch.no_grad():
            logits = self.teacher(x)
        for m in self.watched:
            m.watch = False
        if self.reset_k is not None:
            self.check_shift()
        if not self.update_first:
            out = self.finish(logits)  # before update(): LAME needs this forward's features
        self.remember(x, logits)
        self.update()
        if self.update_first:
            with torch.no_grad():
                out = self.finish(self.teacher(x))
        self.last_logits = out
        return out.argmax(1)


METHODS = {"source": Source, "bn_adapt": BNAdapt, "tent": Tent, "bn_mix": BNMix, "bn_ema": BNEma, "bn_auto": BNAuto,
           "rotta": RoTTA}
