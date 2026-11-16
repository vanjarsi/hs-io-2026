"""Turn low-threshold predictions into the final file for the event scorer.

Steps: (1) optionally take each disease's predictions from a different mask-setting file,
(2) remove duplicate masks (mask IoU > 0.5 with a higher-confidence mask of the same disease),
(3) keep predictions at or above the confidence threshold (one value, or one per disease),
(4) write the scorer's format.

    python make_final.py out.json --preds preds.json --conf 0.16
    python make_final.py out.json --class-file apple_scab=a.json --class-file black_rot=b.json \
        --class-file spot=a.json --class-conf apple_scab=0.185,black_rot=0.19,spot=0.13
"""
import argparse
import json

from evaluate import load_json
from fuse import fuse


def build(preds=None, class_files=None, conf=None, class_conf=None, dedup=0.5):
    if class_files:
        cache, ann = {}, {}
        for cls, f in class_files.items():
            if f not in cache:
                d = load_json(f)
                cache[f] = d.get("annotations", d)
            for name, items in cache[f].items():
                ann.setdefault(name, []).extend(p for p in items if p["label"] == cls)
    else:
        d = load_json(preds)
        ann = d.get("annotations", d)
    res = fuse([ann], thr=dedup, mode="max")
    out = {}
    for name, items in res.items():
        out[name] = [{"id": p["id"], "label": p["label"], "score": p["score"], "points": p["points"]}
                     for p in sorted(items, key=lambda q: -q["score"])
                     if p["score"] >= (class_conf[p["label"]] if class_conf else conf)]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--preds")
    ap.add_argument("--class-file", action="append", default=[])
    ap.add_argument("--conf", type=float)
    ap.add_argument("--class-conf", default="")
    ap.add_argument("--dedup", type=float, default=0.5)
    a = ap.parse_args()
    cf = dict(x.split("=", 1) for x in a.class_file) or None
    cc = {k: float(v) for k, v in (x.split("=") for x in a.class_conf.split(","))} if a.class_conf else None
    out = build(a.preds, cf, a.conf, cc, a.dedup)
    json.dump({"annotation_type": "segmentation", "annotations": out}, open(a.out, "w"))
    print(f"{a.out}: {len(out)} images, {sum(len(v) for v in out.values())} spots")
