"""Remove duplicate predictions: a second, lower-confidence mask on a spot the model already found.

YOLO's built-in NMS compares BOXES at IoU 0.7. For 7-pixel spots a box shifted by a pixel or two
falls under 0.7, so both copies survive, and the copy counts as a false positive. Here a
prediction is dropped when its MASK overlaps a kept, higher-confidence prediction of the same
disease by more than `thr` (mask IoU), or when most of it lies inside one (containment).

    python dedup.py preds.json out.json --thr 0.5 [--contain 0.0] [--conf 0.25] [--agnostic]
"""
import argparse
import json

import numpy as np

from evaluate import _box, _iou, load_json, rast


def dedup_image(items, thr=0.5, contain=0.0, agnostic=False):
    """items: list of prediction dicts for one image. Returns the kept ones (highest confidence first)."""
    order = sorted(items, key=lambda p: -p["score"])
    kept = []
    for p in order:
        m = rast([(q["x"], q["y"]) for q in p["points"]])
        bb = _box(m)
        area = int(m.sum())
        drop = False
        for km, kb, karea, kl in kept:
            if not agnostic and kl != p["label"]:
                continue
            v = _iou(m, bb, km, kb)
            if v > thr:
                drop = True
                break
            if contain and v > 0 and area:
                inter = v * (area + karea) / (1 + v)  # IoU -> intersection area
                if inter / area > contain:
                    drop = True
                    break
        if not drop:
            kept.append((m, bb, area, p["label"]))
            p["_keep"] = True
    out = [p for p in order if p.pop("_keep", False)]
    return out


def dedup(annotations, thr=0.5, contain=0.0, agnostic=False, conf=0.0):
    return {name: dedup_image([p for p in items if p["score"] >= conf], thr, contain, agnostic)
            for name, items in annotations.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    ap.add_argument("out")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--contain", type=float, default=0.0)
    ap.add_argument("--conf", type=float, default=0.0)
    ap.add_argument("--agnostic", action="store_true")
    a = ap.parse_args()
    d = load_json(a.preds)
    ann = d.get("annotations", d)
    before = sum(len(v) for v in ann.values())
    res = dedup(ann, a.thr, a.contain, a.agnostic, a.conf)
    for items in res.values():
        for p in items:
            p.pop("box", None)
    json.dump({"annotation_type": "segmentation", "annotations": res}, open(a.out, "w"))
    print(f"{before} -> {sum(len(v) for v in res.values())} predictions (mask IoU > {a.thr} removed, conf >= {a.conf})")
