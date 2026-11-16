"""Reference YOLO-seg pipeline in numpy + onnxruntime.

It mirrors what the phone app does in JavaScript (and what Ultralytics does in
PyTorch): letterbox -> ONNX model -> confidence filter -> per-class NMS ->
masks (upsample logits, threshold, crop to box). Used by export_model.py to
validate an FP16 conversion and by parity_check.py as a middle column between
"Ultralytics .pt" and "the app in a browser".
"""
import cv2
import numpy as np


def letterbox(img_rgb, size, pad=114):
    """Resize keeping aspect ratio, pad to size x size (Ultralytics LetterBox, auto=False, center=True)."""
    h, w = img_rgb.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    if (nw, nh) != (w, h):
        img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = round(dh - 0.1), round(dw - 0.1)
    out = np.full((size, size, 3), pad, dtype=np.uint8)
    out[top : top + nh, left : left + nw] = img_rgb
    return out, r, left, top


def nms_per_class(boxes, scores, classes, iou_thr, max_det=300):
    order = np.argsort(-scores, kind="stable")[:30000]
    keep = []
    suppressed = np.zeros(len(scores), dtype=bool)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        if len(keep) >= max_det:
            break
        same = (classes == classes[i]) & ~suppressed
        xx1 = np.maximum(boxes[i, 0], boxes[:, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[:, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[:, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[:, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (area[i] + area - inter + 1e-9)
        suppressed |= same & (iou > iou_thr)
    return np.array(keep, dtype=int)


def decode(outputs, size, conf, iou, with_masks=True):
    """outputs: list of arrays from the session. Returns dict(cls, score, box[x1,y1,x2,y2 input px], masks[n,size,size])."""
    det = next(o for o in outputs if o.ndim == 3)[0].astype(np.float32)
    proto = next(o for o in outputs if o.ndim == 4)[0].astype(np.float32)
    if det.shape[0] > det.shape[1]:  # [N, C] layout
        det = det.T
    nm = proto.shape[0]
    nc = det.shape[0] - 4 - nm
    scores_all = det[4 : 4 + nc]
    cls = scores_all.argmax(0)
    score = scores_all.max(0)
    m = score > conf
    cx, cy, w, h = det[0, m], det[1, m], det[2, m], det[3, m]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    cls, score, coef = cls[m], score[m], det[4 + nc :, m].T
    keep = nms_per_class(boxes, score, cls, iou)
    boxes, cls, score, coef = boxes[keep], cls[keep], score[keep], coef[keep]
    res = {"cls": cls, "score": score, "box": boxes}
    if with_masks:
        masks = np.zeros((len(keep), size, size), dtype=bool)
        if len(keep):
            mh, mw = proto.shape[1:]
            logits = (coef @ proto.reshape(nm, -1)).reshape(-1, mh, mw)
            for k in range(len(keep)):
                up = cv2.resize(logits[k], (size, size), interpolation=cv2.INTER_LINEAR) > 0
                x1, y1, x2, y2 = boxes[k]
                xs, ys = np.arange(size), np.arange(size)
                up &= ((xs >= x1) & (xs < x2))[None, :]
                up &= ((ys >= y1) & (ys < y2))[:, None]
                masks[k] = up
        res["masks"] = masks
    return res


def run_onnx(session, img_rgb, size, conf, iou, with_masks=True):
    lb, r, left, top = letterbox(img_rgb, size)
    x = (lb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    inp = session.get_inputs()[0]
    if "float16" in inp.type:
        x = x.astype(np.float16)
    outs = session.run(None, {inp.name: x})
    res = decode(outs, size, conf, iou, with_masks)
    res.update(r=r, left=left, top=top)
    return res
