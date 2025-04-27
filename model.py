#Model

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import batch_norm_momentum, batch_norm_eps, group_size, epsilon  # Import BatchNorm params from config.py

class SamePadConv2d(nn.Conv2d):
    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        dh, dw = self.dilation

        oh = math.ceil(ih / sh)
        ow = math.ceil(iw / sw)
        pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
        pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom])
        return super().forward(x)

# Squeeze-and-Excitation block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        squeezed_channels = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, squeezed_channels, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(squeezed_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale

# MBConv block with SE
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride):
        super(MBConv, self).__init__()
        self.use_residual = (in_channels == out_channels and stride == 1)
        hidden_dim = in_channels * expand_ratio

        self.expand = nn.Sequential(
            SamePadConv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim, momentum=batch_norm_momentum, eps=batch_norm_eps),
            nn.ReLU(inplace=True)
        ) if expand_ratio != 1 else nn.Identity()

        self.depthwise = nn.Sequential(
            SamePadConv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                          groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim, momentum=batch_norm_momentum, eps=batch_norm_eps),
            nn.ReLU(inplace=True)
        )

        self.se = SEBlock(hidden_dim)

        self.project = nn.Sequential(
            SamePadConv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum, eps=batch_norm_eps)
        )

    def forward(self, x):
        identity = x
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            out = out + identity
        return out

# EfficientNet-like model
class EfficientNetCustom(nn.Module):
    def __init__(self, num_classes=37):
        super(EfficientNetCustom, self).__init__()

        self.stem = nn.Sequential(
            SamePadConv2d(3, 32, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(32, momentum=batch_norm_momentum, eps=batch_norm_eps),
            nn.ReLU(inplace=True)
        )

        self.blocks = nn.Sequential(
            # MBConv1, 3x3, stride=1, repeat=1
            MBConv(32, 16, expand_ratio=1, kernel_size=3, stride=1),

            # MBConv6, 3x3, stride=2, repeat=2
            MBConv(16, 32, expand_ratio=6, kernel_size=3, stride=2),
            #MBConv(24, 24, expand_ratio=6, kernel_size=3, stride=1),

            # MBConv6, 5x5, stride=2, repeat=2
            MBConv(32, 64, expand_ratio=6, kernel_size=5, stride=2),
            #MBConv(40, 40, expand_ratio=6, kernel_size=5, stride=1),

            # MBConv6, 3x3, stride=2, repeat=2
            MBConv(64, 128, expand_ratio=6, kernel_size=3, stride=2),
            #MBConv(80, 80, expand_ratio=6, kernel_size=3, stride=1),

            # MBConv6, 5x5, stride=1, repeat=2
            #MBConv(80, 112, expand_ratio=6, kernel_size=5, stride=1),
            #MBConv(112, 112, expand_ratio=6, kernel_size=5, stride=1),

            # MBConv6, 5x5, stride=2, repeat=3
            MBConv(128, 128, expand_ratio=6, kernel_size=5, stride=2),
            #MBConv(112, 112, expand_ratio=6, kernel_size=5, stride=1),
            #MBConv(112, 112, expand_ratio=6, kernel_size=5, stride=1),

            # MBConv6, 3x3, stride=2, repeat=1
            MBConv(128, 128, expand_ratio=6, kernel_size=3, stride=1)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((4,4))

        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(2048, 768),
            nn.ReLU(),
            nn.Linear(768, num_classes)
        )

    def forward(self, x, return_logits=False):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.fc(x)
        #return x

        # Group normalization for Galaxy Zoo
        epsilon = 1e-12
        group_sizes = group_size
        start = 0
        outputs = []
        for g in group_sizes:
            z = x[:, start:start+g]     # raw logits for this question
            z_pos = F.relu(z)           # max(z,0)
            denom = z_pos.sum(dim=1, keepdim=True) + epsilon
            outputs.append(z_pos / denom)
            start += g
        outputs = torch.cat(outputs, dim=1)
        if return_logits:
            return outputs, x
        return outputs
