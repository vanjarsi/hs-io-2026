"""Run a trained YOLO-seg model over a folder of images and save the predictions in the
event scorer's format (the same polygons the Workflow Studio's inference step writes).

    python predict.py --weights ../models/runD_best.pt --images <folder> --imgsz 1024 \
        --conf 0.05 --iou 0.7 --out preds.json [--device mps] [--retina]

Low --conf on purpose: the file keeps every candidate, and the threshold is chosen afterwards.
"""
import argparse
import json
import re
import time
from pathlib import Path

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--images", required=True)
ap.add_argument("--imgsz", type=int, default=1024)
ap.add_argument("--conf", type=float, default=0.05)
ap.add_argument("--iou", type=float, default=0.7)
ap.add_argument("--max-det", type=int, default=300)
ap.add_argument("--device", default="cpu")
ap.add_argument("--retina", action="store_true", help="masks at original image resolution (retina_masks=True)")
ap.add_argument("--agnostic", action="store_true", help="class-agnostic NMS")
ap.add_argument("--flip", default="", choices=["", "h", "v", "hv"], help="test-time flip: predict on the flipped image, then flip the outlines back")
ap.add_argument("--names-file", default="", help="optional text file with one image name per line (subset)")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--out", required=True)
a = ap.parse_args()

model = YOLO(a.weights)
files = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
if a.names_file:
    keep = set(Path(a.names_file).read_text().splitlines())
    files = [p for p in files if p.name in keep]
if a.limit:
    files = files[: a.limit]
import cv2  # noqa: E402

FLIP = {"h": 1, "v": 0, "hv": -1}
out, n, t0 = {}, 0, time.time()
for i in range(0, len(files), a.batch):
    chunk = files[i : i + a.batch]
    srcs = [str(p) for p in chunk]
    if a.flip:
        srcs = [cv2.flip(cv2.imread(str(p)), FLIP[a.flip]) for p in chunk]
    results = model.predict(srcs, imgsz=a.imgsz, conf=a.conf, iou=a.iou, max_det=a.max_det,
                            device=a.device, retina_masks=a.retina, agnostic_nms=a.agnostic, verbose=False)
    for p, r in zip(chunk, results):
        items = []
        if r.masks is not None:
            stem = re.sub(r"[^A-Za-z0-9_-]", "_", p.stem)
            h0, w0 = r.orig_shape
            rx, ry = w0 / r.masks.data.shape[2], h0 / r.masks.data.shape[1]  # size of one mask pixel in image pixels
            for k, seg in enumerate(r.masks.xy):
                if len(seg) < 3:
                    continue
                if "h" in a.flip:
                    seg[:, 0] = w0 - rx - seg[:, 0]
                if "v" in a.flip:
                    seg[:, 1] = h0 - ry - seg[:, 1]
                items.append({
                    "id": f"{stem}-{k}",
                    "label": model.names[int(r.boxes.cls[k])],
                    "score": round(float(r.boxes.conf[k]), 5),
                    "points": [{"x": round(float(x), 2), "y": round(float(y), 2)} for x, y in seg],
                    "box": [round(float(v), 2) for v in r.boxes.xyxy[k]],
                })
        out[p.name] = items
        n += len(items)
    if (i // a.batch) % 25 == 0:
        print(f"  {i + len(chunk)}/{len(files)} images, {n} spots, {time.time() - t0:.0f} s", flush=True)
json.dump({"annotation_type": "segmentation", "annotations": out,
           "meta": {"weights": Path(a.weights).name, "imgsz": a.imgsz, "conf": a.conf, "iou": a.iou,
                    "retina_masks": a.retina, "agnostic_nms": a.agnostic, "flip": a.flip, "seconds": round(time.time() - t0, 1)}},
          open(a.out, "w"))
print(f"saved {a.out}: {len(out)} images, {n} spots, {time.time() - t0:.0f} s ({(time.time() - t0) / max(1, len(files)) * 1000:.0f} ms/image)")
