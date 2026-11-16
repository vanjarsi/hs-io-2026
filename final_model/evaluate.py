"""Score predictions against true labels, at every confidence threshold at once.

Same rule as eval_test.py (and, to within about 0.005, the event's scorer): a predicted mask is
correct when it overlaps an unmatched true mask of the SAME disease with IoU >= 0.5; each true
mask can be matched once; highest-confidence predictions are matched first. Masks are drawn from
the polygons at 256x256 exactly as eval_test.py does. It only looks at mask pairs whose boxes
overlap, so it is fast enough to sweep many settings.

    python evaluate.py preds.json --labels <labels.json | labels folder> [--min-precision 0.80]

Library use: load_labels(), match(), table(), best_at_precision().
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["apple_scab", "black_rot", "spot"]
SIZE = 256


def load_json(path):
    d = json.load(open(path))
    while isinstance(d, str):
        d = json.loads(d)
    return d


def load_labels(src, image_names=None):
    """{image name: YOLO-seg label text}. src = JSON file (test_labels.json) or a folder of .txt files."""
    src = Path(src)
    if src.is_file():
        return load_json(src)
    out = {}
    for name in image_names:
        f = src / (Path(name).stem + ".txt")
        out[name] = f.read_text() if f.exists() else ""
    return out


def rast(points, size=SIZE):
    img = Image.new("1", (size, size), 0)
    if len(points) >= 3:
        ImageDraw.Draw(img).polygon(points, fill=1)
    return np.asarray(img, dtype=bool)


def _box(m):
    ys = np.flatnonzero(m.any(1))
    if not len(ys):
        return None
    xs = np.flatnonzero(m.any(0))
    return xs[0], ys[0], xs[-1] + 1, ys[-1] + 1


def true_masks(text, size=SIZE):
    out = []
    for line in text.splitlines():
        v = line.split()
        if len(v) < 7:
            continue
        xy = [float(t) * size for t in v[1:]]
        m = rast(list(zip(xy[0::2], xy[1::2])), size)
        out.append((int(v[0]), m, _box(m), int(m.sum())))
    return out


def _iou(a, ab, b, bb):
    if ab is None or bb is None:
        return 0.0
    x0, y0, x1, y1 = max(ab[0], bb[0]), max(ab[1], bb[1]), min(ab[2], bb[2]), min(ab[3], bb[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = np.logical_and(a[y0:y1, x0:x1], b[y0:y1, x0:x1]).sum()
    if not inter:
        return 0.0
    return inter / (a.sum() + b.sum() - inter)


def match(annotations, labels, transform=None, iou_min=0.5):
    """Greedy matching. Returns (preds, gts):
    preds: list of dict(img, cls, conf, tp, best_iou, area)   gts: list of dict(img, cls, area, matched_conf)
    transform: optional function(mask) -> mask applied to each predicted mask (e.g. a dilation)."""
    preds, gts = [], []
    for name, items in annotations.items():
        g = true_masks(labels.get(name) or "")
        g0 = len(gts)
        for c, m, bb, area in g:
            gts.append({"img": name, "cls": c, "area": area, "matched_conf": -1.0})
        pm = []
        for p in items:
            m = rast([(q["x"], q["y"]) for q in p["points"]])
            if transform is not None:
                m = transform(m)
            pm.append((CLASSES.index(p["label"]), float(p["score"]), m, _box(m)))
        pm.sort(key=lambda t: -t[1])
        used = [False] * len(g)
        for cls, conf, m, bb in pm:
            best, bj, top = 0.0, -1, 0.0
            for j, (gc, gm, gb, _) in enumerate(g):
                v = _iou(m, bb, gm, gb)
                if gc == cls and v > top:
                    top = v
                if v < iou_min or gc != cls:
                    continue
                if not used[j] and v > best:
                    best, bj = v, j
            if bj >= 0:
                used[bj] = True
                gts[g0 + bj]["matched_conf"] = conf
            preds.append({"img": name, "cls": cls, "conf": conf, "tp": bj >= 0, "best_iou": top, "area": int(m.sum()),
                          "healthy": not g})
    return preds, gts


def prf(preds, gts, thr, cls=None, thr_by_class=None):
    def keep(p):
        t = thr_by_class[p["cls"]] if thr_by_class else thr
        return p["conf"] >= t and (cls is None or p["cls"] == cls)
    ps = [p for p in preds if keep(p)]
    tp = sum(p["tp"] for p in ps)
    nt = sum(1 for g in gts if cls is None or g["cls"] == cls)
    P = tp / len(ps) if ps else 0.0
    R = tp / nt if nt else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {"P": P, "R": R, "F1": F, "tp": tp, "fp": len(ps) - tp, "fn": nt - tp, "n": len(ps)}


def table(preds, gts, thresholds=None):
    thresholds = thresholds if thresholds is not None else [round(0.10 + 0.01 * i, 2) for i in range(41)]
    return [(t, prf(preds, gts, t)) for t in thresholds]


def best_at_precision(preds, gts, min_p=0.80):
    """Lowest threshold (highest recall) whose precision is still >= min_p."""
    best = None
    for t, r in table(preds, gts, [round(0.05 + 0.005 * i, 3) for i in range(131)]):
        if r["P"] >= min_p and (best is None or r["R"] > best[1]["R"]):
            best = (t, r)
    return best


def tiny_recall(gts, thr, max_area=64):
    tiny = [g for g in gts if g["area"] < max_area]
    return sum(g["matched_conf"] >= thr for g in tiny) / max(1, len(tiny)), len(tiny)


def summary(preds, gts, label="", min_p=0.80):
    r30 = prf(preds, gts, 0.30)
    b = best_at_precision(preds, gts, min_p)
    f = max(table(preds, gts), key=lambda x: x[1]["F1"])
    s = f"{label:<34} @0.30 P {r30['P']:.3f} R {r30['R']:.3f} F1 {r30['F1']:.3f} | "
    if b:
        tr, _ = tiny_recall(gts, b[0])
        s += f"P>={min_p:.2f}: conf {b[0]:.3f} P {b[1]['P']:.3f} R {b[1]['R']:.3f} F1 {b[1]['F1']:.3f} tinyR {tr:.3f} | "
    s += f"best F1 {f[1]['F1']:.3f} @ {f[0]:.2f} (P {f[1]['P']:.3f} R {f[1]['R']:.3f})"
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--min-precision", type=float, default=0.80)
    a = ap.parse_args()
    ann = load_json(a.preds)
    ann = ann.get("annotations", ann)
    labels = load_labels(a.labels, list(ann.keys()))
    preds, gts = match(ann, labels)
    print(summary(preds, gts, Path(a.preds).stem, a.min_precision))
    print(" conf    P      R      F1     TP     FP     FN")
    for t, r in table(preds, gts, [0.15, 0.18, 0.20, 0.22, 0.24, 0.25, 0.26, 0.28, 0.30, 0.35, 0.40]):
        print(f" {t:.2f}  {r['P']:.3f}  {r['R']:.3f}  {r['F1']:.3f}  {r['tp']:5d}  {r['fp']:5d}  {r['fn']:5d}")
    for c, n in enumerate(CLASSES):
        r = prf(preds, gts, 0.30, c)
        print(f"  {n:11s} @0.30  P {r['P']:.3f}  R {r['R']:.3f}  F1 {r['F1']:.3f}")
