"""
CRAFT (Character Region Awareness for Text Detection) neural network architecture.

Based on the official CRAFT-pytorch implementation by NAVER Corp.
License: MIT (https://github.com/clovaai/CRAFT-pytorch)

This module implements the CRAFT model architecture including:
- VGG16-BN backbone with dilated convolutions
- U-Net style feature pyramid
- Character region and affinity score map prediction
"""

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def init_weights(modules):
    """Initialize weights using Xavier uniform for Conv2d layers."""
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class VGG16BN(nn.Module):
    """VGG16-BN backbone with dilated convolutions for fc6/fc7."""

    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        from torchvision import models

        vgg_pretrained_features = models.vgg16_bn(
            weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None
        ).features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        for x in range(12):  # conv2_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):  # conv3_3
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 29):  # conv4_3
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(29, 39):  # conv5_3
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        # fc6, fc7 without atrous conv
        self.slice5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1),
        )

        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())

        init_weights(self.slice5.modules())  # no pretrained model for fc6/fc7

        if freeze:
            for param in self.slice1.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h

        VggOutputs = namedtuple(
            "VggOutputs", ["fc7", "relu5_3", "relu4_3", "relu3_2", "relu2_2"]
        )
        return VggOutputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)


class DoubleConv(nn.Module):
    """Double convolution block for U-Net upsampling path."""

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class CRAFTNet(nn.Module):
    """
    CRAFT: Character Region Awareness For Text Detection.

    Predicts character region and affinity score maps using a VGG16-BN
    backbone with U-Net style feature pyramid.

    Output shape: (batch, height/2, width/2, 2) where channels are:
        - channel 0: character region score
        - channel 1: affinity (link) score
    """

    def __init__(self, pretrained: bool = False, freeze: bool = False):
        super().__init__()

        # Base network (VGG16-BN)
        self.basenet = VGG16BN(pretrained, freeze)

        # U-Net upsampling path
        self.upconv1 = DoubleConv(1024, 512, 256)
        self.upconv2 = DoubleConv(512, 256, 128)
        self.upconv3 = DoubleConv(256, 128, 64)
        self.upconv4 = DoubleConv(128, 64, 32)

        num_class = 2  # character region + affinity
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor (batch, 3, H, W), normalized with ImageNet stats.

        Returns:
            y: Score maps (batch, H/2, W/2, 2)
            feature: Feature maps from upconv4 (batch, 32, H/2, W/2)
        """
        # Base network
        sources = self.basenet(x)

        # U-Net decoder
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(
            y, size=sources[2].size()[2:], mode="bilinear", align_corners=False
        )
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(
            y, size=sources[3].size()[2:], mode="bilinear", align_corners=False
        )
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(
            y, size=sources[4].size()[2:], mode="bilinear", align_corners=False
        )
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0, 2, 3, 1), feature


def copy_state_dict(state_dict: dict) -> dict:
    """
    Strip 'module.' prefix from state dict keys (from DataParallel).

    Args:
        state_dict: Model state dictionary potentially with 'module.' prefix.

    Returns:
        Cleaned state dictionary.
    """
    from collections import OrderedDict

    if not state_dict:
        return state_dict

    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict
