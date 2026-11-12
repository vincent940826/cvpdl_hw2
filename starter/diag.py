"""Per-batch diagnostics for analysis runs (`run.py --diag`). Never used for submissions.

For every batch of the stream it records:
- probs: softmax of the logits behind the method's prediction (the method sets `self.last_logits`)
- drift: relative L2 change of the method model's BN affine params (gamma, beta) since the start
- shift_src / shift_prev: per-BN-layer distance between this batch's feature statistics and
  (a) the source statistics, (b) the previous batch's statistics. Measured on a frozen copy of the
  source model, so the signal is the same whichever method is being analysed.
"""
import numpy as np
import torch
import torch.nn as nn

EPS = 1e-5


def sym_kl(m1, v1, m2, v2):
    """Channel-averaged symmetric KL (sum of both directions) between N(m1, v1) and N(m2, v2)."""
    d = (m1 - m2) ** 2
    return (0.5 * ((v1 + d) / v2 + (v2 + d) / v1) - 1).mean().item()


class Recorder:
    def __init__(self, method, source_model):
        self.method = method
        self.ref = source_model.eval().requires_grad_(False)
        bns = [m for m in self.ref.modules() if isinstance(m, nn.BatchNorm2d)]
        self.src = [(m.running_mean.double(), m.running_var.double() + EPS) for m in bns]
        self.cur = [None] * len(bns)
        for i, m in enumerate(bns):
            m.register_forward_hook(self._hook(i))
        self.affine = [p for m in method.model.modules() if isinstance(m, nn.BatchNorm2d)
                       for p in (m.weight, m.bias)]
        self.affine0 = torch.cat([p.detach().flatten() for p in self.affine]).clone()
        self.prev = None
        self.rows = dict(probs=[], drift=[], shift_src=[], shift_prev=[])

    def _hook(self, i):
        def f(module, inp, out):
            h = inp[0].float()
            self.cur[i] = (h.mean((0, 2, 3)).double(), h.var((0, 2, 3), unbiased=False).double() + EPS)
        return f

    def __call__(self, x):
        pred = self.method(x)
        with torch.no_grad():
            self.ref(x)
            cur = list(self.cur)
            self.rows["shift_src"].append([sym_kl(*c, *s) for c, s in zip(cur, self.src)])
            self.rows["shift_prev"].append([np.nan] * len(cur) if self.prev is None
                                           else [sym_kl(*c, *p) for c, p in zip(cur, self.prev)])
            self.prev = cur
            logits = getattr(self.method, "last_logits", None)
            self.rows["probs"].append(logits.float().softmax(1).cpu().numpy() if logits is not None
                                      else np.full((len(x), 20), np.nan, dtype=np.float32))
            now = torch.cat([p.detach().flatten() for p in self.affine])
            self.rows["drift"].append(((now - self.affine0).norm() / self.affine0.norm()).item())
        return pred

    def save(self, path):
        extra = {k: np.array(getattr(self.method, k)) for k in ("alphas", "resets") if hasattr(self.method, k)}
        np.savez_compressed(path, probs=np.concatenate(self.rows["probs"]).astype(np.float32),
                            drift=np.array(self.rows["drift"]), shift_src=np.array(self.rows["shift_src"]),
                            shift_prev=np.array(self.rows["shift_prev"]), **extra)
