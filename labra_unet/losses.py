"""Focal-Tversky composite loss (focal loss + Tversky loss) for imbalanced lesion pixels."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction="none")
        return (self.alpha * (1 - torch.exp(-ce)) ** self.gamma * ce).mean()


class TverskyLoss(nn.Module):
    """beta > alpha penalizes false-negative lesion pixels more than false positives."""

    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0, n_cls=3):
        super().__init__()
        self.alpha, self.beta, self.smooth, self.n_cls = alpha, beta, smooth, n_cls

    def forward(self, logits, target):
        p = F.softmax(logits, dim=1)
        t = F.one_hot(target, self.n_cls).permute(0, 3, 1, 2).float()
        tp = (p * t).sum(dim=(2, 3))
        fp = ((1 - t) * p).sum(dim=(2, 3))
        fn = (t * (1 - p)).sum(dim=(2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal, self.tversky = FocalLoss(), TverskyLoss()

    def forward(self, logits, target):
        return self.focal(logits, target) + self.tversky(logits, target)
