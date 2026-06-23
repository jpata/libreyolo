import numpy as np
import torch


def decode_detections(det_logits: torch.Tensor, conf_thresh: float = 0.25, nms_thresh: float = 0.45) -> list:
    if len(det_logits.shape) == 4:
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

    boxes = []
    for y in range(h_grid):
        for x in range(w_grid):
            obj_score = pred_obj[y, x].item()
            if obj_score < conf_thresh:
                continue
            dx = pred_xy[0, y, x].item()
            dy = pred_xy[1, y, x].item()
            dw = pred_wh[0, y, x].item()
            dh = pred_wh[1, y, x].item()
            cx_cell = (x + dx) / w_grid
            cy_cell = (y + dy) / h_grid
            w_box = dw / w_grid
            h_box = dh / h_grid
            x1 = cx_cell - w_box / 2.0
            y1 = cy_cell - h_box / 2.0
            x2 = cx_cell + w_box / 2.0
            y2 = cy_cell + h_box / 2.0
            if num_thing_classes > 1:
                class_id = int(torch.argmax(pred_cls[:, y, x]).item())
                class_score = pred_cls[class_id, y, x].item()
            else:
                class_id = 0
                class_score = pred_cls[0, y, x].item()
            boxes.append([x1, y1, x2, y2, obj_score * class_score, float(class_id)])

    if not boxes:
        return []

    boxes_np = np.array(boxes, dtype=np.float64)
    x1_arr = boxes_np[:, 0]
    y1_arr = boxes_np[:, 1]
    x2_arr = boxes_np[:, 2]
    y2_arr = boxes_np[:, 3]
    scores_arr = boxes_np[:, 4]

    areas = (x2_arr - x1_arr) * (y2_arr - y1_arr)
    order = scores_arr.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1_arr[i], x1_arr[order[1:]])
        yy1 = np.maximum(y1_arr[i], y1_arr[order[1:]])
        xx2 = np.minimum(x2_arr[i], x2_arr[order[1:]])
        yy2 = np.minimum(y2_arr[i], y2_arr[order[1:]])
        w_inter = np.maximum(0.0, xx2 - xx1)
        h_inter = np.maximum(0.0, yy2 - yy1)
        inter = w_inter * h_inter
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        inds = np.where(iou <= nms_thresh)[0]
        order = order[inds + 1]

    return boxes_np[keep].tolist()