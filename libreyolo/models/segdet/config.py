from dataclasses import dataclass, field
from typing import ClassVar, Dict


@dataclass
class SegDetSizeConfig:
    backbone: str
    fpn_channels: int
    conv_dims: int
    input_size: int


SIZE_CONFIGS: Dict[str, SegDetSizeConfig] = {
    "n": SegDetSizeConfig(backbone="resnet9", fpn_channels=64, conv_dims=64, input_size=320),
    "s": SegDetSizeConfig(backbone="resnet9", fpn_channels=128, conv_dims=128, input_size=416),
    "m": SegDetSizeConfig(backbone="resnet18", fpn_channels=128, conv_dims=128, input_size=544),
    "l": SegDetSizeConfig(backbone="resnet18", fpn_channels=256, conv_dims=256, input_size=640),
}

NUM_CONVS = 2