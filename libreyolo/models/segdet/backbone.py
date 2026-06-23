import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_norm=True, use_act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not use_norm)
        self.bn = nn.BatchNorm2d(out_channels) if use_norm else nn.Identity()
        self.act = nn.ReLU(inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, stride=stride)
        self.conv2 = ConvBlock(out_channels, out_channels, use_act=False)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + self.shortcut(x))


class ResNetBackbone(nn.Module):
    def __init__(self, width_list=(64, 64, 128, 256, 512), depth_list=(2, 2, 2, 2), final_stage_stride=2, num_dino_channels=None):
        super().__init__()
        self.width_list = list(width_list)
        self.final_stage_stride = final_stage_stride
        if num_dino_channels is not None:
            self.width_list[-1] = num_dino_channels

        self.stem = nn.Sequential(
            nn.Conv2d(3, self.width_list[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.width_list[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        in_c = self.width_list[0]
        self.stage1 = self._make_stage(in_c, self.width_list[1], depth_list[0], stride=1)
        self.stage2 = self._make_stage(self.width_list[1], self.width_list[2], depth_list[1], stride=2)
        self.stage3 = self._make_stage(self.width_list[2], self.width_list[3], depth_list[2], stride=2)
        self.stage4 = self._make_stage(self.width_list[3], self.width_list[4], depth_list[3], stride=final_stage_stride)

    def _make_stage(self, in_c, out_c, depth, stride):
        layers = [ResBlock(in_c, out_c, stride=stride)]
        for _ in range(1, depth):
            layers.append(ResBlock(out_c, out_c, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = {}
        x = self.stem(x)
        out["stage0"] = x
        out["stage1"] = self.stage1(x)
        out["stage2"] = self.stage2(out["stage1"])
        out["stage3"] = self.stage3(out["stage2"])
        out["stage4"] = self.stage4(out["stage3"])
        out["stage_final"] = out["stage4"]
        return out


def resnet9(final_stage_stride=2, num_dino_channels=None):
    return ResNetBackbone(
        width_list=[64, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 1],
        final_stage_stride=final_stage_stride,
        num_dino_channels=num_dino_channels,
    )


def resnet18(final_stage_stride=2, num_dino_channels=None):
    return ResNetBackbone(
        width_list=[64, 64, 128, 256, 512],
        depth_list=[2, 2, 2, 2],
        final_stage_stride=final_stage_stride,
        num_dino_channels=num_dino_channels,
    )


def resnet18_stride16(num_dino_channels=384):
    return ResNetBackbone(
        width_list=[64, 64, 128, 256, 512],
        depth_list=[2, 2, 2, 2],
        final_stage_stride=1,
        num_dino_channels=num_dino_channels,
    )