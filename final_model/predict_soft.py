"""Predict with the exported ONNX model and write one predictions file per mask setting.

YOLO draws a spot's mask where the mask probability is above 0.5, then crops it to the spot's box.
Run D's near-misses are mostly masks that are too small, so this script also writes masks cut at
lower probabilities (0.4, 0.3, ...) and, optionally, cropped to a slightly enlarged box. The
network runs once per image; only the last step differs between settings.

    python predict_soft.py --onnx model_fp32.onnx --images <folder> --imgsz 1024 --out-prefix out/test_D
        [--thr 0.5,0.4,0.3,0.2] [--expand 0,0.2] [--conf 0.10] [--iou 0.7] [--names-file subset.txt]

Setting thr=0.5, expand=0 reproduces Ultralytics / the Workflow Studio exactly.
"""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phone_app"))
from onnx_ref import letterbox, nms_per_class  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True)
ap.add_argument("--images", required=True)
ap.add_argument("--imgsz", type=int, default=1024)
ap.add_argument("--conf", type=float, default=0.10)
ap.add_argument("--iou", type=float, default=0.7)
ap.add_argument("--thr", default="0.5,0.4,0.3,0.2")
ap.add_argument("--expand", default="0,0.2")
ap.add_argument("--names", default="apple_scab,black_rot,spot")
ap.add_argument("--names-file", default="")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out-prefix", required=True)
a = ap.parse_args()

NAMES = a.names.split(",")
THR = [float(t) for t in a.thr.split(",")]
EXP = [float(e) for e in a.expand.split(",")]
S = a.imgsz
sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
files = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
if a.names_file:
    keep = set(Path(a.names_file).read_text().splitlines())
    files = [p for p in files if p.name in keep]
if a.limit:
    files = files[: a.limit]

variants = {(t, e): {} for t in THR for e in EXP}
t0 = time.time()
for fi, p in enumerate(files):
    img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    lb, r, left, top = letterbox(img, S)
    x = (lb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    outs = sess.run(None, {inp: x})
    det = next(o for o in outs if o.ndim == 3)[0]
    proto = next(o for o in outs if o.ndim == 4)[0]
    if det.shape[0] > det.shape[1]:
        det = det.T
    nm, mh, mw = proto.shape
    nc = det.shape[0] - 4 - nm
    sc = det[4 : 4 + nc]
    cls_all, score_all = sc.argmax(0), sc.max(0)
    m = score_all > a.conf
    cx, cy, w, h = det[0, m], det[1, m], det[2, m], det[3, m]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    cls, score, coef = cls_all[m], score_all[m], det[4 + nc :, m].T
    k = nms_per_class(boxes, score, cls, a.iou)
    boxes, cls, score, coef = boxes[k], cls[k], score[k], coef[k]
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", p.stem)
    items = {v: [] for v in variants}
    if len(k):
        logits = (coef @ proto.reshape(nm, -1)).reshape(-1, mh, mw)
        for d in range(len(k)):
            up = cv2.resize(logits[d], (S, S), interpolation=cv2.INTER_LINEAR)
            x1, y1, x2, y2 = boxes[d]
            bw, bh = x2 - x1, y2 - y1
            for e in EXP:
                ex1, ey1, ex2, ey2 = x1 - e * bw, y1 - e * bh, x2 + e * bw, y2 + e * bh
                X0, X1 = max(0, math.ceil(ex1)), min(S, math.ceil(ex2))
                Y0, Y1 = max(0, math.ceil(ey1)), min(S, math.ceil(ey2))
                if X1 <= X0 or Y1 <= Y0:
                    continue
                crop = up[Y0:Y1, X0:X1]
                for t in THR:
                    mk = (crop > math.log(t / (1 - t))).astype(np.uint8)
                    cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not cs:
                        continue
                    c = max(cs, key=lambda q: (cv2.contourArea(q), len(q))).reshape(-1, 2)
                    if len(c) < 3:
                        continue
                    items[(t, e)].append({
                        "id": f"{stem}-{d}", "label": NAMES[int(cls[d])], "score": round(float(score[d]), 5),
                        "points": [{"x": round((float(px) + X0 - left) / r, 2), "y": round((float(py) + Y0 - top) / r, 2)} for px, py in c],
                    })
    for v in variants:
        variants[v][p.name] = items[v]
    if fi % 100 == 0:
        print(f"  {fi + 1}/{len(files)} images, {time.time() - t0:.0f} s", flush=True)

for (t, e), ann in variants.items():
    out = f"{a.out_prefix}_t{t:g}_e{e:g}.json"
    json.dump({"annotation_type": "segmentation", "annotations": ann,
               "meta": {"onnx": Path(a.onnx).name, "imgsz": S, "conf": a.conf, "iou": a.iou, "mask_threshold": t, "box_expand": e}},
              open(out, "w"))
    print(f"saved {out}: {sum(len(v) for v in ann.values())} spots")
print(f"done in {time.time() - t0:.0f} s ({(time.time() - t0) / max(1, len(files)) * 1000:.0f} ms/image)")
