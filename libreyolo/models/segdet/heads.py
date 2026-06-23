from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1, use_norm=True):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=not use_norm)
        self.dw_bn = nn.BatchNorm2d(in_channels) if use_norm else nn.Identity()
        self.dw_act = nn.ReLU(inplace=True)

        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=not use_norm)
        self.pw_bn = nn.BatchNorm2d(out_channels) if use_norm else nn.Identity()
        self.pw_act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw_act(self.dw_bn(self.dw(x)))
        x = self.pw_act(self.pw_bn(self.pw(x)))
        return x


class FPN(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1, bias=False) for c in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            DWSeparableConv(out_channels, out_channels, kernel_size=3) for _ in in_channels_list
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        laterals = [lat(f) for lat, f in zip(self.lateral_convs, features)]
        for i in range(len(laterals) - 1, 0, -1):
            h_prev, w_prev = laterals[i - 1].shape[-2:]
            h_i, w_i = laterals[i].shape[-2:]
            upsampled = F.interpolate(laterals[i], scale_factor=(h_prev / h_i, w_prev / w_i), mode="nearest")
            laterals[i - 1] = laterals[i - 1] + upsampled
        return [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]


class SingleHead(nn.Module):
    def __init__(self, in_channels: int, conv_dims: int, num_convs: int):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.append(DWSeparableConv(
                in_channels=in_channels if i == 0 else conv_dims,
                out_channels=conv_dims,
                kernel_size=3,
            ))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)


class MultiLevelHead(nn.Module):
    def __init__(self, in_channels: int, conv_dims: int, num_convs: int, num_levels: int):
        super().__init__()
        self.heads = nn.ModuleList([
            SingleHead(in_channels, conv_dims, num_convs) for _ in range(num_levels)
        ])

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        target_h, target_w = features[0].shape[-2:]
        upsampled_outputs = []
        for head, f in zip(self.heads, features):
            out = head(f)
            h, w = out.shape[-2:]
            if h != target_h or w != target_w:
                out = F.interpolate(out, scale_factor=(target_h / h, target_w / w), mode="nearest")
            upsampled_outputs.append(out)
        combined = upsampled_outputs[0]
        for i in range(1, len(upsampled_outputs)):
            combined = combined + upsampled_outputs[i]
        return combined


class PositionHead(nn.Module):
    def __init__(self, fpn_channels: int, conv_dims: int, num_convs: int, num_classes: int, num_levels: int):
        super().__init__()
        self.position_head = MultiLevelHead(fpn_channels, conv_dims, num_convs, num_levels)
        self.out = nn.Conv2d(conv_dims, num_classes, kernel_size=3, padding=1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        x = self.position_head(feats)
        return self.out(x)


class DetectionHead(nn.Module):
    def __init__(self, fpn_channels: int, conv_dims: int, num_convs: int, num_thing_classes: int, num_levels: int):
        super().__init__()
        self.detection_head = MultiLevelHead(fpn_channels, conv_dims, num_convs, num_levels)
        self.out = nn.Conv2d(conv_dims, 4 + 1 + num_thing_classes, kernel_size=1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        x = self.detection_head(feats)
        return self.out(x)