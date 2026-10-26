"""
LabRA-UNet: U-Net for CIELAB leaf images with residual blocks and attention gates.

Every encoder stage, the bottleneck and every decoder stage is a residual block.
Each skip connection passes through an attention gate before it is concatenated
with the upsampled decoder features.
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Two 3 x 3 conv-BN layers plus a shortcut (identity, or 1 x 1 conv-BN when the width changes)."""

    def __init__(self, i, o):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(i, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                  nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o))
        self.skip = nn.Identity() if i == o else nn.Sequential(nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.body(x) + self.skip(x))


class AttentionGate(nn.Module):
    """Attention gate on a skip connection (Oktay et al., 2018).

    The encoder feature map x and the decoder gating signal g are projected with 1 x 1
    convolutions and added; ReLU, a 1 x 1 convolution and a sigmoid give the attention
    coefficients alpha in [0, 1], which multiply x element-wise.
    """

    def __init__(self, f_g, f_x, f_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1), nn.BatchNorm2d(f_int))
        self.W_x = nn.Sequential(nn.Conv2d(f_x, f_int, 1), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        alpha = self.psi(self.relu(self.W_g(g) + self.W_x(x)))
        return x * alpha


class LabRAUNet(nn.Module):
    """Input: 3-channel CIELAB image. Output: logits for background, BP and LC."""

    def __init__(self, n_cls=3, widths=(64, 128, 256, 512, 1024)):
        super().__init__()
        w = widths
        self.pool = nn.MaxPool2d(2)
        self.enc = nn.ModuleList([ResidualBlock(3, w[0]), ResidualBlock(w[0], w[1]),
                                  ResidualBlock(w[1], w[2]), ResidualBlock(w[2], w[3])])
        self.bottleneck = ResidualBlock(w[3], w[4])
        levels = (3, 2, 1, 0)
        self.up = nn.ModuleList([nn.ConvTranspose2d(w[k + 1], w[k], 2, 2) for k in levels])
        self.att = nn.ModuleList([AttentionGate(w[k], w[k], w[k] // 2) for k in levels])
        self.dec = nn.ModuleList([ResidualBlock(2 * w[k], w[k]) for k in levels])
        self.head = nn.Conv2d(w[0], n_cls, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else self.pool(x))
            skips.append(x)
        d = self.bottleneck(self.pool(x))
        for up, att, dec, skip in zip(self.up, self.att, self.dec, reversed(skips)):
            g = up(d)                                    # decoder gating signal
            d = dec(torch.cat((att(g, skip), g), 1))     # refined encoder features + decoder features
        return self.head(d)
