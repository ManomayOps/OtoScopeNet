import torch
import torch.nn as nn
import numpy as np

def channel_shuffle(x, groups):
    """Performs channel shuffle operation."""
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x

class ECSA(nn.Module):
    """Efficient Channel and Spatial Attention (ECSA) Block."""
    def __init__(self, in_channels, gamma=2, b=1):
        super(ECSA, self).__init__()
        t = int(abs((np.log2(in_channels) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Main forward pass for ECSA
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        s = x.mean(dim=1, keepdim=True)
        s = self.sigmoid(s)
        return x * y.expand_as(x) * s.expand_as(x)

class GhostModule(nn.Module):
    """GhostNet Module."""
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        init_channels = int(np.ceil(out_channels / ratio))
        new_channels = init_channels * (ratio - 1)
        
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

    def forward(self, x):
        # Main forward pass for GhostModule
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]

class ShuffleBlock(nn.Module):
    """The main building block of the network."""
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ShuffleBlock, self).__init__()
        mid_channels = out_channels // 2
        self.downsample = downsample
        
        self.main_path = nn.Sequential(
            GhostModule(in_channels, mid_channels, kernel_size=1, relu=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2 if downsample else 1, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            GhostModule(mid_channels, out_channels, kernel_size=1, relu=False),
            ECSA(out_channels)
        )
        
        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Main forward pass for ShuffleBlock
        shortcut = self.shortcut(x) if self.downsample else x
        main = self.main_path(x)
        out = self.relu(main + shortcut)
        return channel_shuffle(out, 2)

class EdgeOtoscopeNet(nn.Module):
    """The complete model architecture."""
    def __init__(self, num_classes=5):
        super(EdgeOtoscopeNet, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.stage1 = self._make_stage(16, 64, num_blocks=2, downsample=True)
        self.stage2 = self._make_stage(64, 128, num_blocks=3, downsample=True)
        self.stage3 = self._make_stage(128, 256, num_blocks=4, downsample=True)
        
        self.head = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def _make_stage(self, in_channels, out_channels, num_blocks, downsample):
        layers = []
        layers.append(ShuffleBlock(in_channels, out_channels, downsample=downsample))
        for _ in range(1, num_blocks):
            layers.append(ShuffleBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Main forward pass for EdgeOtoscopeNet
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
        return x