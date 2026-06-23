import torch
import torchvision


def decode_detections(det_logits: torch.Tensor, conf_thresh: float = 0.25, nms_thresh: float = 0.45) -> list:
    if det_logits.dim() == 4:
        pred = det_logits[0]
    else:
        pred = det_logits
    c, h_grid, w_grid = pred.shape
    num_thing_classes = c - 5

    pred_xy = pred[0:2].sigmoid()
    pred_wh = torch.exp(torch.clamp(pred[2:4], min=-5, max=5))
    pred_obj = pred[4].sigmoid()
    pred_cls = pred[5:]

    if num_thing_classes > 1:
        pred_cls = pred_cls.softmax(dim=0)
    else:
        pred_cls = pred_cls.sigmoid()

    y_grid = torch.arange(h_grid, device=det_logits.device)
    x_grid = torch.arange(w_grid, device=det_logits.device)
    gy, gx = torch.meshgrid(y_grid, x_grid, indexing="ij")

    cx_cell = (gx + pred_xy[0]) / w_grid
    cy_cell = (gy + pred_xy[1]) / h_grid
    w_box = pred_wh[0] / w_grid
    h_box = pred_wh[1] / h_grid

    x1 = cx_cell - w_box / 2.0
    y1 = cy_cell - h_box / 2.0
    x2 = cx_cell + w_box / 2.0
    y2 = cy_cell + h_box / 2.0

    if num_thing_classes > 1:
        class_scores, class_ids = pred_cls.max(dim=0)
    else:
        class_scores = pred_cls[0]
        class_ids = torch.zeros_like(class_scores, dtype=torch.long)

    scores = pred_obj * class_scores
    obj_mask = pred_obj >= conf_thresh
    if not obj_mask.any():
        return []

    x1_v = x1[obj_mask]
    y1_v = y1[obj_mask]
    x2_v = x2[obj_mask]
    y2_v = y2[obj_mask]
    scores_v = scores[obj_mask]
    class_ids_v = class_ids[obj_mask]

    keep = torchvision.ops.batched_nms(
        torch.stack([x1_v, y1_v, x2_v, y2_v], dim=1),
        scores_v,
        class_ids_v,
        nms_thresh,
    )
    keep = keep[scores_v[keep].argsort(descending=True)]

    boxes = torch.stack(
        [x1_v[keep], y1_v[keep], x2_v[keep], y2_v[keep], scores_v[keep], class_ids_v[keep].float()], dim=1
    )
    return boxes.tolist()