"""A lightweight 3D U-Net implementation (PyTorch).

Usage:
  from tracking.unet3d import UNet3D
  model = UNet3D(in_channels=1, out_channels=1, base_filters=32)
"""
from typing import Tuple
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if necessary
        if x.shape[-3:] != skip.shape[-3:]:
            diffz = skip.size(-3) - x.size(-3)
            diffy = skip.size(-2) - x.size(-2)
            diffx = skip.size(-1) - x.size(-1)
            x = nn.functional.pad(x, [diffx//2, diffx - diffx//2, diffy//2, diffy - diffy//2, diffz//2, diffz - diffz//2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_filters: int = 32):
        if nn is None:
            raise RuntimeError('PyTorch is required to use UNet3D')
        super().__init__()
        f = base_filters
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = Down(f, f*2)
        self.enc3 = Down(f*2, f*4)
        self.enc4 = Down(f*4, f*8)

        self.center = ConvBlock(f*8, f*16)

        self.up4 = Up(f*16, f*8)
        self.up3 = Up(f*8, f*4)
        self.up2 = Up(f*4, f*2)
        self.up1 = Up(f*2, f)

        self.final = nn.Conv3d(f, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.enc1(x)
        c2 = self.enc2(c1)
        c3 = self.enc3(c2)
        c4 = self.enc4(c3)
        cen = self.center(c4)
        u4 = self.up4(cen, c4)
        u3 = self.up3(u4, c3)
        u2 = self.up2(u3, c2)
        u1 = self.up1(u2, c1)
        out = self.final(u1)
        return out
