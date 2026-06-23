from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.image_loader import ImageInput

from ..base.model import BaseModel
from ...training.ddp_spawn import ddp_aware
from .backbone import resnet9, resnet18
from .config import SIZE_CONFIGS, NUM_CONVS
from .heads import FPN, PositionHead, DetectionHead
from .postprocess import postprocess_detect, postprocess_semantic
from .utils import preprocess_image

logger = logging.getLogger(__name__)


_TRAIN_DEFAULTS = None


def _get_train_defaults():
    global _TRAIN_DEFAULTS
    if _TRAIN_DEFAULTS is None:
        from .trainer import SegDetTrainConfig
        _TRAIN_DEFAULTS = SegDetTrainConfig()
    return _TRAIN_DEFAULTS


_ARCH_KEYS = frozenset({
    "backbone.stem.0.weight",
    "fpn.lateral_convs.0.weight",
    "position_head.position_head.heads.0.convs.0.dw.weight",
})


class LibreSegDetModel(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        fpn_channels: int = 128,
        conv_dims: int = 128,
        num_classes: int = 2,
        num_thing_classes: int = 1,
        num_convs: int = NUM_CONVS,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_thing_classes = num_thing_classes

        if backbone == "resnet9":
            self.backbone = resnet9()
        elif backbone == "resnet18":
            self.backbone = resnet18()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        fpn_in = self.backbone.width_list[1:5]
        num_levels = len(fpn_in)
        self.fpn = FPN(in_channels_list=fpn_in, out_channels=fpn_channels)
        self.position_head = PositionHead(fpn_channels, conv_dims, num_convs, num_classes, num_levels)
        self.detection_head = DetectionHead(fpn_channels, conv_dims, num_convs, num_thing_classes, num_levels)

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        fpn_feats = [feats["stage1"], feats["stage2"], feats["stage3"], feats["stage4"]]
        fpn_outs = self.fpn(fpn_feats)
        stuff_logits_low = self.position_head(fpn_outs)
        semantic_logits = F.interpolate(stuff_logits_low, scale_factor=4.0, mode="bilinear", align_corners=False)
        detection_logits = self.detection_head(fpn_outs)
        return {"semantic_logits": semantic_logits, "detection_logits": detection_logits}


class LibreSegDet(BaseModel):
    FAMILY = "segdet"
    FILENAME_PREFIX = "LibreSegDet"
    INPUT_SIZES: ClassVar[Dict[str, int]] = {k: v.input_size for k, v in SIZE_CONFIGS.items()}
    SUPPORTED_TASKS = ("detect", "semantic")
    DEFAULT_TASK = "detect"
    TTA_ENABLED = False

    @classmethod
    def _get_train_config(cls):
        from .trainer import SegDetTrainConfig
        return SegDetTrainConfig

    @classmethod
    def _normalize_state_dict_keys(cls, weights_dict: dict) -> dict:
        result = {}
        for k, v in weights_dict.items():
            key = k
            if key.startswith("module."):
                key = key[7:]
            if key.startswith("model."):
                key = key[6:]
            result[key] = v
        return result

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        keys = set(cls._normalize_state_dict_keys(weights_dict))
        return _ARCH_KEYS.issubset(keys)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        wd = cls._normalize_state_dict_keys(weights_dict)
        stem_w = wd.get("backbone.stem.0.weight")
        if stem_w is None:
            return None
        stem_c = stem_w.shape[0]
        stage3_w = wd.get("backbone.stage3.0.convs.0.conv.weight")
        if stage3_w is None:
            return None
        stage3_c = stage3_w.shape[0]
        if stem_c == 32 and stage3_c <= 128:
            return "n"
        if stem_c == 64 and stage3_c <= 128:
            return "s"
        if stem_c == 64 and stage3_c <= 256:
            return "m"
        if stem_c == 64 and stage3_c > 256:
            return "l"
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        wd = cls._normalize_state_dict_keys(weights_dict)
        out_w = wd.get("position_head.out.weight")
        if out_w is not None:
            return out_w.shape[0]
        return None

    @classmethod
    def detect_num_thing_classes(cls, weights_dict: dict) -> Optional[int]:
        wd = cls._normalize_state_dict_keys(weights_dict)
        out_w = wd.get("detection_head.out.weight")
        if out_w is not None:
            c = out_w.shape[0]
            return max(c - 5, 0)
        return None

    @classmethod
    def get_download_url(cls, _filename: str) -> Optional[str]:
        return None

    def __init__(
        self,
        model_path: str | Path | None = None,
        size: str = "s",
        nb_classes: int = 2,
        device: str = "auto",
        task: str | None = None,
        num_thing_classes: int = 1,
        **kwargs: Any,
    ):
        self.num_thing_classes = num_thing_classes
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=task,
            **kwargs,
        )
        if isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))

    def _init_model(self) -> nn.Module:
        cfg = SIZE_CONFIGS[self.size]
        return LibreSegDetModel(
            backbone=cfg.backbone,
            fpn_channels=cfg.fpn_channels,
            conv_dims=cfg.conv_dims,
            num_classes=self.nb_classes,
            num_thing_classes=self.num_thing_classes,
            num_convs=NUM_CONVS,
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "fpn": self.model.fpn,
            "position_head": self.model.position_head,
            "detection_head": self.model.detection_head,
        }

    @staticmethod
    def _get_preprocess_numpy():
        return None

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Any, Tuple[int, int], float]:
        isz = input_size or self._get_input_size()
        tensor_np, pil_img, orig_size, ratio = preprocess_image(image, isz, color_format)
        tensor = torch.from_numpy(tensor_np)
        return tensor, pil_img, orig_size, ratio

    def _forward(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        ratio: float = 1.0,
        **kwargs: Any,
    ) -> Dict:
        if self.task == "semantic":
            return postprocess_semantic(output, conf_thres, iou_thres, original_size, max_det=max_det, ratio=ratio)
        return postprocess_detect(output, conf_thres, iou_thres, original_size, max_det=max_det, ratio=ratio)

    def _load_weights(self, model_path: str) -> None:
        from ...utils.serialization import load_untrusted_torch_file

        loaded = load_untrusted_torch_file(model_path, map_location="cpu", context="segdet")
        if isinstance(loaded, dict) and "model" in loaded:
            state_dict = loaded["model"]
        elif isinstance(loaded, dict) and "state_dict" in loaded:
            state_dict = loaded["state_dict"]
            state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
        else:
            state_dict = loaded

        state_dict = self._normalize_state_dict_keys(state_dict)
        strict = self._strict_loading()
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if strict:
            if missing:
                logger.warning("Missing keys: %s", missing)
            if unexpected:
                logger.warning("Unexpected keys: %s", unexpected)

    def _rebuild_for_new_classes(self, new_nb_classes: int):
        if new_nb_classes == self.nb_classes:
            return
        old_state = self.model.state_dict()
        old_nc = self.nb_classes
        self.nb_classes = new_nb_classes
        self.model = self._init_model()

        new_state = self.model.state_dict()
        for k in old_state:
            if k in new_state and old_state[k].shape == new_state[k].shape:
                new_state[k] = old_state[k]
        self.model.load_state_dict(new_state, strict=False)

    def _strict_loading(self) -> bool:
        return False

    def _restore_after_training(self, results: dict) -> None:
        checkpoint = None
        for key in ("best_checkpoint", "last_checkpoint"):
            path = results.get(key)
            if path and Path(path).exists():
                checkpoint = str(path)
                break
        if checkpoint is not None:
            self.model_path = checkpoint
            self._load_weights(checkpoint)
        self.model.to(self.device).eval()

    @ddp_aware()
    def train(
        self,
        data: str,
        *,
        epochs: int = _get_train_defaults().epochs,
        batch: int = _get_train_defaults().batch,
        imgsz: int = _get_train_defaults().imgsz,
        lr0: float = _get_train_defaults().lr0,
        optimizer: str = _get_train_defaults().optimizer,
        device: str = "",
        workers: int = _get_train_defaults().workers,
        seed: int = _get_train_defaults().seed,
        project: str = _get_train_defaults().project,
        name: str = _get_train_defaults().name,
        exist_ok: bool = _get_train_defaults().exist_ok,
        resume: bool = _get_train_defaults().resume,
        amp: bool = _get_train_defaults().amp,
        patience: int = _get_train_defaults().patience,
        allow_download_scripts: bool = False,
        pretrained: bool | str | Path | None = None,
        callbacks=None,
        loggers=None,
        **kwargs,
    ) -> dict:
        from .trainer import SegDetTrainer
        from ...data import load_data_config

        try:
            data_config = load_data_config(
                data,
                autodownload=True,
                allow_scripts=allow_download_scripts,
            )
            data = data_config.get("yaml_file", data)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load dataset config '{data}': {e}")

        yaml_nc = data_config.get("nc")
        yaml_names = data_config.get("names")
        yaml_num_thing = data_config.get("num_thing_classes", 1)

        if yaml_nc is None and yaml_names is not None:
            yaml_nc = len(yaml_names)
        if yaml_nc is not None:
            yaml_nc = int(yaml_nc)

        if yaml_nc is not None and yaml_nc != self.nb_classes:
            self._rebuild_for_new_classes(yaml_nc)

        if yaml_names is not None:
            if isinstance(yaml_names, list):
                yaml_names = {i: n for i, n in enumerate(yaml_names)}
            self.names = self._sanitize_names(yaml_names, self.nb_classes)

        if resume and pretrained:
            raise ValueError("pretrained transfer cannot be combined with resume=True.")

        if pretrained:
            transfer_weights: str | Path
            if pretrained is True:
                transfer_weights = self._default_transfer_weights_name()
            else:
                transfer_weights = pretrained
            stats = self._load_transfer_weights(transfer_weights)
            logger.info(
                "Loaded %d transfer tensors from %s; skipped %d incompatible tensors.",
                stats["loaded"],
                transfer_weights,
                stats["skipped"],
            )

        if seed >= 0:
            import random
            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if str(device).lower() not in ("cpu", "mps") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        trainer_kwargs = dict(
            model=self.model,
            wrapper_model=self,
            size=self.size,
            num_classes=self.nb_classes,
            num_thing_classes=yaml_num_thing,
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            lr0=lr0,
            optimizer=optimizer.lower(),
            device=device if device else "auto",
            workers=workers,
            seed=seed,
            project=project,
            name=name,
            exist_ok=exist_ok,
            resume=resume,
            amp=amp,
            patience=patience,
            allow_download_scripts=allow_download_scripts,
            callbacks=callbacks,
            loggers=loggers,
            **kwargs,
        )
        trainer = SegDetTrainer(**trainer_kwargs)

        if resume:
            if not self.model_path:
                raise ValueError(
                    "resume=True requires a checkpoint. Load one first: "
                    "model = LibreSegDet('path/to/last.pt', size='s'); model.train(data=..., resume=True)"
                )
            trainer.setup()
            trainer.resume(str(self.model_path))

        results = trainer.train()

        self._restore_after_training(results)

        return results