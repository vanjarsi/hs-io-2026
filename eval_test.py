"""Score a run's test-set predictions against the true labels.

The Workflow Studio's inference step writes annotations.json (one mask outline
per predicted spot, in pixels) but no precision/recall: the test images are
scored by nobody. This does it, the way the ICICLE rubric grades a run.

A predicted mask counts as correct (true positive) when it overlaps an unmatched
true mask of the SAME disease with IoU >= 0.5. Each true mask can be matched once,
highest-confidence predictions first (the standard detection rule).

Usage:
    python eval_test.py annotations.json labels.json [--size 256] [--out results.json]

labels.json = {"<image name>": "<YOLO-seg label text>"} (class x1 y1 x2 y2 ..., normalised).
"""
import argparse
import json

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["apple_scab", "black_rot", "spot"]
THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]


def load(path):
    d = json.load(open(path))
    while isinstance(d, str):  # files saved from the browser may be JSON-encoded twice
        d = json.loads(d)
    return d


def mask(points, size):
    img = Image.new("1", (size, size), 0)
    if len(points) >= 3:
        ImageDraw.Draw(img).polygon(points, fill=1)
    return np.asarray(img, dtype=bool)


def true_masks(label_text, size):
    out = []
    for line in label_text.splitlines():
        v = line.split()
        if len(v) < 7:
            continue
        xy = [float(t) * size for t in v[1:]]
        out.append((CLASSES[int(v[0])], mask(list(zip(xy[0::2], xy[1::2])), size)))
    return out


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    return inter / (np.logical_or(a, b).sum() or 1)


def score(annotations, labels, size=256, iou_min=0.5):
    """Per-prediction match records, so any threshold can be scored afterwards."""
    preds_out, n_true = [], {c: 0 for c in CLASSES}
    healthy_images = 0
    for name, preds in annotations.items():
        gts = true_masks(labels.get(name) or "", size)
        if not gts:
            healthy_images += 1
        for c, _ in gts:
            n_true[c] += 1
        pm = [(p["label"], p["score"], mask([(q["x"], q["y"]) for q in p["points"]], size))
              for p in preds]
        pm.sort(key=lambda t: -t[1])
        used = [False] * len(gts)
        for cls, conf, m in pm:
            best, best_j, wrong_class = 0.0, -1, False
            for j, (gc, gm) in enumerate(gts):
                v = iou(m, gm)
                if v < iou_min:
                    continue
                if gc != cls:
                    wrong_class = True
                    continue
                if not used[j] and v > best:
                    best, best_j = v, j
            if best_j >= 0:
                used[best_j] = True
            preds_out.append({"cls": cls, "conf": conf, "tp": best_j >= 0,
                              "wrong_class": best_j < 0 and wrong_class,
                              "healthy": not gts})
    return preds_out, n_true, healthy_images


def summarize(preds, n_true, thr):
    rows = {}
    for c in CLASSES + ["all"]:
        ps = [p for p in preds if p["conf"] >= thr and (c == "all" or p["cls"] == c)]
        tp = sum(p["tp"] for p in ps)
        nt = sum(n_true.values()) if c == "all" else n_true[c]
        prec = tp / len(ps) if ps else 0.0
        rec = tp / nt if nt else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows[c] = {"precision": prec, "recall": rec, "F1": f1, "tp": tp,
                   "fp": len(ps) - tp, "fn": nt - tp, "predicted": len(ps), "true": nt}
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("annotations")
    ap.add_argument("labels")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out")
    a = ap.parse_args()
    ann = load(a.annotations)
    preds, n_true, healthy = score(ann.get("annotations", ann), load(a.labels), a.size)
    report = {"healthy_images": healthy, "n_true": n_true,
              "wrong_class": sum(p["wrong_class"] for p in preds),
              "fp_on_healthy": sum(p["healthy"] for p in preds),
              "by_threshold": {str(t): summarize(preds, n_true, t) for t in THRESHOLDS}}
    for t in THRESHOLDS:
        r = report["by_threshold"][str(t)]["all"]
        print(f"conf>={t:.2f}  P {r['precision']:.3f}  R {r['recall']:.3f}  F1 {r['F1']:.3f}  "
              f"(TP {r['tp']}, FP {r['fp']}, FN {r['fn']})")
    for c in CLASSES:
        r = report["by_threshold"]["0.25"][c]
        print(f"  {c:11s} @0.25  P {r['precision']:.3f}  R {r['recall']:.3f}  F1 {r['F1']:.3f}  "
              f"(true {r['true']}, predicted {r['predicted']})")
    print("healthy images:", healthy, "| predictions on healthy leaves:", report["fp_on_healthy"],
          "| right place, wrong disease:", report["wrong_class"])
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)
