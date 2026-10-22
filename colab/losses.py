
import torch, torch.nn as nn, torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0): super().__init__(); self.a=alpha; self.g=gamma
    def forward(self, x, y):
        ce = F.cross_entropy(x, y, reduction='none')
        return (self.a * (1-torch.exp(-ce))**self.g * ce).mean()

class TverskyLoss(nn.Module):
    # Recall을 높이기 위해 False Negative에 가중치(beta=0.7)를 더 줌
    def __init__(self, alpha=0.3, beta=0.7, smooth=1):
        super().__init__(); self.a=alpha; self.b=beta; self.s=smooth
    def forward(self, x, y):
        x = F.softmax(x, dim=1)
        y = F.one_hot(y, 3).permute(0,3,1,2).float()

        # Batch, Class 별 합계
        tp = (x * y).sum(dim=(2, 3))
        fp = ((1 - y) * x).sum(dim=(2, 3))
        fn = (y * (1 - x)).sum(dim=(2, 3))

        tversky = (tp + self.s) / (tp + self.a * fp + self.b * fn + self.s)
        return 1 - tversky.mean()

class ComboLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss()
        self.tversky = TverskyLoss() # Dice 대신 Tversky 사용

    def forward(self, x, y):
        # Focal(분류 정확도) + Tversky(영역 채우기 강화)
        return self.focal(x, y) + self.tversky(x, y)
