from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from ...training.trainer import BaseTrainer
from ...training.config import TrainConfig
from ...data import load_data_config

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SegDetTrainConfig(TrainConfig):
    loss_seg_weight: float = 1.0
    loss_obj_weight: float = 1.0
    loss_box_weight: float = 5.0
    loss_cls_weight: float = 1.0
    num_thing_classes: int = 1
    eval_interval: int = 1


class SegDetTrainer(BaseTrainer):
    artifact_model_families = ("segdet",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_mIoU = 0.0

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return SegDetTrainConfig

    def get_model_family(self) -> str:
        return "segdet"

    def get_model_tag(self) -> str:
        return f"LibreSegDet-{self.config.size}"

    def create_transforms(self):
        from ...data.transforms import Compose, Normalize, ToTensor, Resize

        target_size = (self.config.imgsz, self.config.imgsz)
        return Compose([
            Resize(target_size, interpolation="bilinear"),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]), None

    def create_scheduler(self, iters_per_epoch: int):
        from ...training.scheduler import LinearLRScheduler
        return LinearLRScheduler(
            lr=self.config.lr0,
            iters_per_epoch=iters_per_epoch,
            total_epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs,
            warmup_lr_start=self.config.warmup_lr_start,
            min_lr_ratio=self.config.min_lr_ratio,
        )

    def get_loss_components(self, outputs: Dict) -> Dict[str, torch.Tensor]:
        return {
            "loss_seg": outputs.get("loss_seg", torch.tensor(0.0)),
            "loss_obj": outputs.get("loss_obj", torch.tensor(0.0)),
            "loss_box": outputs.get("loss_box", torch.tensor(0.0)),
            "loss_cls": outputs.get("loss_cls", torch.tensor(0.0)),
        }

    def _setup_data(self):
        cfg = load_data_config(self.config.data) if isinstance(self.config.data, str) else self.config.data
        self.data_cfg = cfg

        nc = int(cfg.get("nc", 2))
        self.num_thing_classes = int(cfg.get("num_thing_classes", 1))

        if hasattr(self.model, "nb_classes"):
            self.model.nb_classes = nc

        train_path = cfg.get("train", "images")
        val_path = cfg.get("val", "images")

        self.train_loader = self._build_loader(train_path, cfg, augment=True)
        self.val_loader = self._build_loader(val_path, cfg, augment=False)

    def _build_loader(self, img_dir, cfg, augment=False):
        from torch.utils.data import DataLoader

        dataset_root = Path(cfg["path"])
        label_dir = cfg.get("labels_dir", "labels")
        mask_dir = cfg.get("masks_dir")
        if label_dir:
            label_dir = str(dataset_root / label_dir)
        if mask_dir:
            mask_dir = str(dataset_root / mask_dir)

        dataset = SegDetDataset(
            img_dir=img_dir,
            label_dir=label_dir,
            mask_dir=mask_dir,
            nc=int(cfg.get("nc", 2)),
            num_thing_classes=self.num_thing_classes,
            target_size=self.config.imgsz,
            augment=augment,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch,
            shuffle=augment,
            num_workers=self.config.workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch):
        imgs = torch.stack([b["data"] for b in batch])
        labels = torch.stack([b["label"] for b in batch])
        instances = torch.stack([b["instance"] for b in batch])
        # Pack label + instance into single 2-channel int64 tensor
        targets = torch.stack([labels, instances.to(torch.int64)], dim=1)
        img_infos = [b.get("image_path", "") for b in batch]
        img_ids = list(range(len(batch)))
        return imgs, targets, img_infos, img_ids

    def on_forward(self, imgs, targets, polygons=None):
        batch = {
            "data": imgs,
            "label": targets[:, 0],
            "instance": targets[:, 1],
        }
        outputs = self.model(imgs)
        loss_dict = self.loss(outputs, batch)
        outputs["total_loss"] = loss_dict["loss"]
        outputs["loss_seg"] = loss_dict["loss_seg"]
        outputs["loss_obj"] = loss_dict["loss_obj"]
        outputs["loss_box"] = loss_dict["loss_box"]
        outputs["loss_cls"] = loss_dict["loss_cls"]
        return outputs

    def loss(self, predictions: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        sem_logits = predictions["semantic_logits"]
        det_logits = predictions["detection_logits"]

        gt_semantic = batch["label"]
        gt_instance = batch["instance"]

        loss_seg = F.cross_entropy(sem_logits, gt_semantic)

        grid_h, grid_w = det_logits.shape[-2:]
        gt_obj, gt_box, gt_cls, box_mask = self._prepare_targets(batch, (grid_h, grid_w))

        pred_obj = det_logits[:, 4, :, :]
        loss_obj_unreduced = F.binary_cross_entropy_with_logits(pred_obj, gt_obj, reduction="none")
        pos = loss_obj_unreduced[gt_obj == 1]
        neg = loss_obj_unreduced[gt_obj == 0]
        loss_obj = pos.mean() + neg.mean() if pos.numel() > 0 else neg.mean()

        pred_xy = det_logits[:, :2, :, :].sigmoid()
        pred_wh = det_logits[:, 2:4, :, :]
        pred_boxes = torch.cat([pred_xy, pred_wh], dim=1)

        if box_mask.any():
            pred_flat = pred_boxes.permute(0, 2, 3, 1)[box_mask]
            gt_flat = gt_box.permute(0, 2, 3, 1)[box_mask]
            loss_box = F.l1_loss(pred_flat, gt_flat, reduction="mean")
        else:
            loss_box = torch.tensor(0.0, device=det_logits.device)

        pred_cls = det_logits[:, 5:, :, :]
        if box_mask.any() and self.num_thing_classes > 0:
            pred_cls_flat = pred_cls.permute(0, 2, 3, 1)[box_mask]
            gt_cls_flat = gt_cls[box_mask]
            loss_cls = F.cross_entropy(pred_cls_flat, gt_cls_flat, reduction="mean")
        else:
            loss_cls = torch.tensor(0.0, device=det_logits.device)

        w_seg = getattr(self.config, "loss_seg_weight", 1.0)
        w_obj = getattr(self.config, "loss_obj_weight", 1.0)
        w_box = getattr(self.config, "loss_box_weight", 5.0)
        w_cls = getattr(self.config, "loss_cls_weight", 1.0)

        total = w_seg * loss_seg + w_obj * loss_obj + w_box * loss_box + w_cls * loss_cls

        return {
            "loss": total,
            "loss_seg": loss_seg,
            "loss_obj": loss_obj,
            "loss_box": loss_box,
            "loss_cls": loss_cls,
        }

    def _prepare_targets(self, batch, grid_size):
        images = batch["data"]
        instances = batch["instance"]
        gt_semantic = batch["label"]
        b, c, h_img, w_img = images.shape
        h_grid, w_grid = grid_size
        device = images.device

        gt_obj = torch.zeros((b, h_grid, w_grid), device=device)
        gt_box = torch.zeros((b, 4, h_grid, w_grid), device=device)
        gt_cls = torch.zeros((b, h_grid, w_grid), dtype=torch.long, device=device)
        box_mask = torch.zeros((b, h_grid, w_grid), dtype=torch.bool, device=device)

        for i in range(b):
            inst = instances[i]
            unique_ids = torch.unique(inst)
            unique_ids = unique_ids[unique_ids != 0]

            for inst_id in unique_ids:
                pixels = (inst == inst_id)
                pos = torch.where(pixels)
                if pos[0].numel() == 0:
                    continue
                ymin, xmin = pos[0].min().float(), pos[1].min().float()
                ymax, xmax = pos[0].max().float(), pos[1].max().float()
                w_norm = (xmax - xmin) / w_img
                h_norm = (ymax - ymin) / h_img
                cx_norm = (xmin + xmax) / (2.0 * w_img)
                cy_norm = (ymin + ymax) / (2.0 * h_img)

                if w_norm <= 1e-4 or h_norm <= 1e-4:
                    continue

                grid_x = int(cx_norm * w_grid)
                grid_y = int(cy_norm * h_grid)
                grid_x = min(max(grid_x, 0), w_grid - 1)
                grid_y = min(max(grid_y, 0), h_grid - 1)

                # Get class ID from semantic label
                class_ids_in_instance = gt_semantic[i][pixels]
                sem_class_id = torch.mode(class_ids_in_instance).values.item()
                if not (1 <= sem_class_id <= 7):
                    continue
                yolo_cls = sem_class_id - 1

                gt_obj[i, grid_y, grid_x] = 1.0
                box_mask[i, grid_y, grid_x] = True
                gt_box[i, 0, grid_y, grid_x] = cx_norm * w_grid - grid_x
                gt_box[i, 1, grid_y, grid_x] = cy_norm * h_grid - grid_y
                gt_box[i, 2, grid_y, grid_x] = torch.log(w_norm * w_grid + 1e-6)
                gt_box[i, 3, grid_y, grid_x] = torch.log(h_norm * h_grid + 1e-6)
                gt_cls[i, grid_y, grid_x] = yolo_cls

        return gt_obj, gt_box, gt_cls, box_mask

    def _run_validation(
        self, epoch: int, *, save_plots: bool | None = None
    ) -> Optional[Dict[str, Any]]:
        try:
            from libreyolo.validation import (
                DetectionValidator,
                SemanticValidator,
                ValidationConfig,
            )

            logger.info(f"Running validation for epoch {epoch + 1}")

            is_final_epoch = self._is_final_epoch(epoch)
            val_save_plots = (
                bool(save_plots)
                if save_plots is not None
                else bool(getattr(self.config, "save_plots", False)) and is_final_epoch
            )
            val_save_dir = (
                str(self.save_dir / "val") if val_save_plots else None
            )

            val_config = ValidationConfig(
                data=self.config.data,
                batch_size=self.config.batch,
                imgsz=self.config.imgsz,
                conf_thres=0.001,
                iou_thres=0.65,
                device=str(self.device),
                half=self.config.amp and self.device.type == "cuda",
                verbose=False,
                num_workers=0,
                save_plots=val_save_plots,
                save_dir=val_save_dir,
            )

            if self.wrapper_model is None:
                logger.error(
                    "Validation requires wrapper_model to be provided to trainer"
                )
                return None

            from ...training.trainer import unwrap_model
            eval_pytorch_model = (
                self.ema_model.ema if self.ema_model else unwrap_model(self.model)
            )
            original_model = self.wrapper_model.model
            self.wrapper_model.model = eval_pytorch_model

            try:
                # 1. Run DetectionValidator
                det_validator = DetectionValidator(model=self.wrapper_model, config=val_config)
                det_results = det_validator.run()

                # 2. Run SemanticValidator
                sem_validator = SemanticValidator(model=self.wrapper_model, config=val_config)
                sem_results = sem_validator.run()
            finally:
                self.wrapper_model.model = original_model

            results = {}
            results.update(det_results)
            results.update(sem_results)

            raw_metrics = self._scalar_mapping(results)
            best_key = getattr(self, "best_metric_key", "metrics/mAP50-95")
            best_metric = raw_metrics.get(
                best_key, raw_metrics.get("metrics/mAP50-95", 0.0)
            )
            mAP50 = raw_metrics.get(
                "metrics/mAP50", raw_metrics.get("metrics/mAP50(B)", 0.0)
            )
            miou = raw_metrics.get("metrics/mIoU", 0.0)

            metrics = {
                "mAP50": mAP50,
                "mAP50_95": best_metric,
                "best_metric": best_metric,
                "best_metric_key": best_key,
                "best_mIoU": miou,
                "metrics": raw_metrics,
            }

            logger.debug(
                f"Extracted metrics: mAP50={metrics['mAP50']:.4f}, mAP50_95={metrics['mAP50_95']:.4f}, mIoU={metrics['best_mIoU']:.4f}"
            )
            logger.info(
                "Validation - mAP50: %.4f, mAP50-95: %.4f, mIoU: %.4f",
                metrics["mAP50"],
                metrics["mAP50_95"],
                metrics["best_mIoU"],
            )
            return metrics

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            import traceback
            logger.debug(f"Validation traceback:\n{traceback.format_exc()}")
            return None

    def _update_best_state(
        self, epoch: int, val_metrics: Optional[Dict[str, Any]]
    ) -> bool:
        if val_metrics is None:
            return False
        miou = val_metrics.get("best_mIoU", 0.0)
        if miou > self.best_mIoU:
            self.best_mIoU = miou
        return super()._update_best_state(epoch, val_metrics)

    def _build_train_results(self) -> Dict[str, Any]:
        results = super()._build_train_results()
        results["best_mIoU"] = self.best_mIoU
        return results


class SegDetDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, mask_dir, nc, num_thing_classes, target_size=640, augment=False):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir) if label_dir else None
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.nc = nc
        self.num_thing_classes = num_thing_classes
        self.target_size = (target_size, target_size)
        self.augment = augment

        self.img_paths = sorted(list(self.img_dir.glob("*.png")) + list(self.img_dir.glob("*.jpg")))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        from PIL import Image
        import numpy as np

        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize(self.target_size, Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        img_t = torch.from_numpy(img_np.transpose(2, 0, 1))

        instance_t = torch.zeros(self.target_size, dtype=torch.int32)
        if self.label_dir:
            label_path = self.label_dir / (img_path.stem + ".txt")
            if label_path.exists():
                instance_id = 1
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:5])
                            x1 = int((cx - w / 2) * self.target_size[0])
                            y1 = int((cy - h / 2) * self.target_size[1])
                            x2 = int((cx + w / 2) * self.target_size[0])
                            y2 = int((cy + h / 2) * self.target_size[1])
                            instance_t[y1:y2, x1:x2] = instance_id
                            instance_id += 1

        sem_t = None
        if self.mask_dir:
            mask_path = self.mask_dir / (img_path.stem + ".png")
            if mask_path.exists():
                mask = Image.open(mask_path)
                mask = mask.resize(self.target_size, Image.NEAREST)
                sem_t = torch.from_numpy(np.array(mask, dtype=np.int64))

        if sem_t is None:
            sem_t = torch.zeros(self.target_size, dtype=torch.int64)

        return {"data": img_t, "label": sem_t, "instance": instance_t, "image_path": str(img_path)}