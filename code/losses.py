"""Classification losses and batch-wise Mixup/CutMix with hard-label mixing."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def build_criterion(kind="ce", **kw):
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        if kw.get("weight") is None:
            raise ValueError("ce_weighted needs train-derived class weights")
        return nn.CrossEntropyLoss(weight=torch.as_tensor(kw["weight"], dtype=torch.float32))
    raise ValueError(f"Unknown loss: {kind}")


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        if not 0 <= smoothing < 1:
            raise ValueError("smoothing must be in [0,1)")
        self.smoothing = smoothing

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and nonnegative")
        self.gamma = gamma
        alpha = None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32)
        if alpha is not None and (alpha.ndim != 1 or not torch.isfinite(alpha).all() or (alpha <= 0).any()):
            raise ValueError("alpha must be a positive finite class-weight vector")
        self.register_buffer("alpha", alpha)

    def forward(self, logits, target):
        log_pt = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        losses = -(1 - log_pt.exp()).pow(self.gamma) * log_pt
        if self.alpha is None:
            return losses.mean()
        weights = self.alpha[target]
        # Same reduction as weighted CE, so gamma=0 also matches weighted CE.
        return (losses * weights).sum() / weights.sum()


def class_weights(counts, beta=0.0):
    counts = torch.as_tensor(counts, dtype=torch.float64)
    if counts.shape != (9,) or not torch.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError("counts must contain nine positive train-class counts")
    if not 0 <= beta < 1:
        raise ValueError("beta must be in [0,1)")
    if beta == 0:
        weights = 1 / counts
    else:
        weights = (1 - beta) / (-torch.expm1(counts * math.log(beta)))
    return (weights / weights.mean()).float()


def mix_batch(x, y, alpha=1.0, mode="cutmix"):
    if mode not in ("mixup", "cutmix") or not math.isfinite(alpha) or alpha < 0:
        raise ValueError("mode must be mixup/cutmix and alpha finite, nonnegative")
    if alpha == 0:
        return x, (y, y, 1.0)
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(len(x), device=x.device)
    if mode == "mixup":
        mixed = lam * x + (1 - lam) * x[perm]
    else:
        h, w = x.shape[-2:]
        cut_h, cut_w = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
        cy, cx = int(torch.randint(h, (1,))), int(torch.randint(w, (1,)))
        y1, y2 = max(0, cy - cut_h // 2), min(h, cy + cut_h - cut_h // 2)
        x1, x2 = max(0, cx - cut_w // 2), min(w, cx + cut_w - cut_w // 2)
        mixed = x.clone()
        mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)
    return mixed, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
