import torch
import torch.nn.functional as F

from .decoder import decode_detections


def postprocess_detect(output, conf_thres, iou_thres, original_size, max_det=300, ratio=1.0, classes=None):
    sem_logits = output["semantic_logits"]
    det_logits = output["detection_logits"]

    orig_w, orig_h = original_size
    boxes = decode_detections(det_logits, conf_thresh=conf_thres, nms_thresh=iou_thres)

    if not boxes:
        return {
            "boxes": torch.empty((0, 4)),
            "scores": torch.empty(0),
            "classes": torch.empty(0, dtype=torch.int),
            "num_detections": 0,
        }

    box_t = torch.tensor([b[:4] for b in boxes], dtype=torch.float32)
    box_t[:, [0, 2]] *= orig_w
    box_t[:, [1, 3]] *= orig_h

    score_t = torch.tensor([b[4] for b in boxes], dtype=torch.float32)
    cls_t = torch.tensor([int(b[5]) for b in boxes], dtype=torch.int)

    if max_det > 0 and len(box_t) > max_det:
        box_t = box_t[:max_det]
        score_t = score_t[:max_det]
        cls_t = cls_t[:max_det]

    if classes is not None:
        mask = torch.zeros(len(cls_t), dtype=torch.bool)
        for cid in classes:
            mask |= cls_t == cid
        box_t = box_t[mask]
        score_t = score_t[mask]
        cls_t = cls_t[mask]

    return {
        "boxes": box_t,
        "scores": score_t,
        "classes": cls_t,
        "num_detections": len(box_t),
    }


def postprocess_semantic(output, conf_thres, iou_thres, original_size, max_det=300, ratio=1.0, classes=None):
    sem_logits = output["semantic_logits"]
    orig_w, orig_h = original_size
    logits = F.interpolate(sem_logits.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    class_map = logits.argmax(dim=1)[0].cpu()
    return {"semantic": class_map, "num_detections": 0}