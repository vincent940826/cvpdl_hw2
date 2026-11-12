"""TTA methods. Add your own here and register it in METHODS.

A method is a callable: method(x) -> LongTensor of labels in {0..19} for the normalized batch x.
It may update its own state (BN statistics, parameters, memory, ...) as the stream goes on.
"""
import torch
import torch.nn as nn


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


METHODS = {"source": Source, "bn_adapt": BNAdapt, "tent": Tent}
