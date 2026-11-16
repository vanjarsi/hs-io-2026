"""Fuse several prediction files (different views of the same image, or different models) into one.

Predictions of the same disease whose masks overlap (mask IoU > thr) are treated as the same
spot. The fused confidence is the sum of the member confidences divided by the number of
sources, so a spot that every source finds keeps its confidence and a spot that only one source
finds is marked down. A second mask from the SAME source on an already-found spot is a duplicate
and is dropped. With a single source this is plain duplicate removal.

    python fuse.py out.json a.json b.json ... [--thr 0.5] [--mode avg|max|noisyor]
"""
import argparse
import json

from evaluate import _box, _iou, load_json, rast


def fuse(sources, thr=0.5, mode="avg", weights=None, min_score=0.0):
    n = len(sources)
    weights = weights or [1.0] * n
    names = list(sources[0].keys())
    out = {}
    for name in names:
        cand = []
        for si, src in enumerate(sources):
            for p in src.get(name, []):
                if p["score"] >= min_score:
                    cand.append((p["score"], si, p))
        cand.sort(key=lambda t: -t[0])
        clusters = []  # [mask, box, label, members{source: score}, representative prediction]
        for score, si, p in cand:
            m = rast([(q["x"], q["y"]) for q in p["points"]])
            bb = _box(m)
            best, bi = thr, -1
            for ci, c in enumerate(clusters):
                if c[2] != p["label"]:
                    continue
                v = _iou(m, bb, c[0], c[1])
                if v > best:
                    best, bi = v, ci
            if bi < 0:
                clusters.append([m, bb, p["label"], {si: score}, p])
            elif si not in clusters[bi][3]:
                clusters[bi][3][si] = score
            # else: same source, same spot -> duplicate, dropped
        items = []
        for m, bb, label, members, rep in clusters:
            if mode == "avg":
                s = sum(members[k] * weights[k] for k in members) / sum(weights)
            elif mode == "max":
                s = max(members.values())
            else:  # noisy-or
                q = 1.0
                for v in members.values():
                    q *= 1 - v
                s = 1 - q
            items.append({"id": rep.get("id", ""), "label": label, "score": round(float(s), 5), "points": rep["points"],
                          "votes": len(members)})
        out[name] = items
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--mode", default="avg")
    a = ap.parse_args()
    srcs = []
    for f in a.inputs:
        d = load_json(f)
        srcs.append(d.get("annotations", d))
    res = fuse(srcs, a.thr, a.mode)
    json.dump({"annotation_type": "segmentation", "annotations": res}, open(a.out, "w"))
    print(f"fused {len(srcs)} files -> {sum(len(v) for v in res.values())} predictions")
