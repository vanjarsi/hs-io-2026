"""Parity check: does the phone app find the same spots as Ultralytics?

Runs the bundled sample images through three pipelines and compares them:
  1. Ultralytics in Python on the .pt file   model.predict(imgsz, conf, iou)
  2. the exported ONNX file in Python        onnxruntime + onnx_ref.py
  3. the real app in headless Chrome         tools/run_app.js (onnxruntime-web)
and, when available, against the predictions the Workflow Studio's own
inference step saved for the same test images (runB_test_annotations.json).

    python parity_check.py --pt models/runB_best.pt
        [--images healthy_1.jpg,scab_1.jpg,rust_1.jpg]   default: every bundled sample
        [--annotations ../.playwright-mcp/runB_test_annotations.json]
        [--url http://127.0.0.1:8000/]     the app must be served (python3 serve.py)
        [--app-json file.json]             reuse a saved run of tools/run_app.js

Two detections "match" when they have the same class and their masks overlap
with IoU >= 0.5 (the rule eval_test.py uses), matched greedily from the
highest confidence. Writes parity/parity_results.json.
"""
import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from onnx_ref import run_onnx  # noqa: E402


def load_json(path):
    d = json.loads(Path(path).read_text())
    while isinstance(d, str):  # files saved from the browser may be JSON-encoded twice
        d = json.loads(d)
    return d


def match(a, b, iou_min=0.5):
    """a, b: lists of dict(cls, score, mask[bool HxW]). Returns matched pairs (i, j, iou)."""
    pairs, used = [], set()
    for i in sorted(range(len(a)), key=lambda k: -a[k]["score"]):
        best, bj = iou_min, None
        for j in range(len(b)):
            if j in used or b[j]["cls"] != a[i]["cls"]:
                continue
            inter = np.logical_and(a[i]["mask"], b[j]["mask"]).sum()
            if not inter:
                continue
            v = inter / np.logical_or(a[i]["mask"], b[j]["mask"]).sum()
            if v >= best:
                best, bj = v, j
        if bj is not None:
            used.add(bj)
            pairs.append((i, bj, float(best)))
    return pairs


def compare(a, b):
    pairs = match(a, b)
    dconf = [abs(a[i]["score"] - b[j]["score"]) for i, j, _ in pairs]
    return {
        "n_a": len(a),
        "n_b": len(b),
        "matched": len(pairs),
        "mean_mask_iou": round(float(np.mean([p[2] for p in pairs])), 4) if pairs else None,
        "mean_abs_conf_diff": round(float(np.mean(dconf)), 4) if pairs else None,
        "max_abs_conf_diff": round(float(np.max(dconf)), 4) if pairs else None,
        "classes_a": sorted({d["cls"] for d in a}),
        "classes_b": sorted({d["cls"] for d in b}),
    }


def counts_ok(n_a, n_b):
    return abs(n_a - n_b) <= max(1, round(0.10 * max(n_a, n_b))) if (n_a or n_b) else True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", default=None, help="the .pt the ONNX file was exported from (default: source_model in config.json)")
    ap.add_argument("--images", default=None)
    ap.add_argument("--annotations", default=str(HERE.parent / ".playwright-mcp" / "runB_test_annotations.json"))
    ap.add_argument("--url", default="http://127.0.0.1:8000/")
    ap.add_argument("--app-json", default=None)
    ap.add_argument("--out", default=str(HERE / "parity" / "parity_results.json"))
    args = ap.parse_args()

    cfg = json.loads((HERE / "model" / "config.json").read_text())
    S, conf, iou = cfg["imgsz"], cfg["conf"], cfg["iou"]
    names = [c["name"] for c in cfg["classes"]]
    samples = json.loads((HERE / "samples" / "samples.json").read_text())
    by_file = {s["file"]: s for s in samples}
    files = args.images.split(",") if args.images else [s["file"] for s in samples]
    pt = Path(args.pt) if args.pt else HERE / cfg["source_model"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. the app (first, so a missing server fails fast)
    if args.app_json:
        app = load_json(args.app_json)
    else:
        app_file = out_path.parent / "app_results.json"
        cmd = ["node", str(HERE / "tools" / "run_app.js"), "--url", args.url, "--out", str(app_file),
               "--images", ",".join("samples/" + f for f in files), "--repeat", "3",
               "--opts", json.dumps({"match": False, "crop": False})]  # plain letterbox, as Ultralytics does
        print("Running the app in headless Chrome ...")
        subprocess.run(cmd, check=True, cwd=HERE)
        app = load_json(app_file)

    # 1. Ultralytics on the .pt
    from ultralytics import YOLO
    import ultralytics

    model = YOLO(str(pt))
    # 2. the ONNX file in Python
    import onnxruntime as ort

    sess = ort.InferenceSession(str(HERE / "model" / cfg["model_file"]), providers=["CPUExecutionProvider"])

    ann = None
    if args.annotations and Path(args.annotations).exists():
        ann = load_json(args.annotations)["annotations"]

    rows, report = [], {}
    for f in files:
        img_bgr = cv2.imread(str(HERE / "samples" / f))
        h, w = img_bgr.shape[:2]

        # rect=False: square letterbox, as the fixed-shape ONNX model needs (no effect on square images)
        r = model.predict(img_bgr, imgsz=S, conf=conf, iou=iou, rect=False, verbose=False)[0]
        ul = []
        if r.masks is not None:
            md = r.masks.data.cpu().numpy().astype(bool)
            for k in range(len(r.boxes)):
                m = md[k]
                if m.shape != (S, S):  # rectangular inference pads less than the app: centre it in S x S
                    full = np.zeros((S, S), bool)
                    oy, ox = (S - m.shape[0]) // 2, (S - m.shape[1]) // 2
                    full[oy : oy + m.shape[0], ox : ox + m.shape[1]] = m
                    m = full
                ul.append({"cls": int(r.boxes.cls[k]), "score": float(r.boxes.conf[k]), "mask": m})

        o = run_onnx(sess, cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), S, conf, iou)
        onx = [{"cls": int(o["cls"][k]), "score": float(o["score"][k]), "mask": o["masks"][k]} for k in range(len(o["score"]))]

        a = app["results"][f]
        ap_ = []
        for d in a["dets"]:
            m = np.zeros((S, S), bool)
            mk = d["mask"]
            if mk["w"] and mk["h"]:
                patch = np.frombuffer(base64.b64decode(mk["b64"]), dtype=np.uint8).reshape(mk["h"], mk["w"]).astype(bool)
                m[mk["y"] : mk["y"] + mk["h"], mk["x"] : mk["x"] + mk["w"]] = patch
            ap_.append({"cls": d["cls"], "score": d["score"], "mask": m})

        entry = {
            "image": f,
            "source_image": by_file.get(f, {}).get("source"),
            "true_spots": by_file.get(f, {}).get("true_spots"),
            "counts": {"ultralytics_pt": len(ul), "onnx_python": len(onx), "app_browser": len(ap_)},
            "app_vs_ultralytics": compare(ap_, ul),
            "app_vs_onnx_python": compare(ap_, onx),
            "onnx_python_vs_ultralytics": compare(onx, ul),
            "app_timing_ms": {k: round(v, 1) for k, v in a["timing"].items()},
        }
        studio_n = None
        src = by_file.get(f, {}).get("source")
        if ann is not None and src in ann:
            k = S / max(h, w)
            st = []
            for p in ann[src]:
                if p["score"] < conf:
                    continue
                pts = np.array([[q["x"] * k, q["y"] * k] for q in p["points"]], dtype=np.int32)
                m = np.zeros((S, S), np.uint8)
                if len(pts) >= 3:
                    cv2.fillPoly(m, [pts], 1)
                st.append({"cls": names.index(p["label"]), "score": float(p["score"]), "mask": m.astype(bool)})
            entry["app_vs_studio_annotations"] = compare(ap_, st)
            studio_n = len(st)
            entry["counts"]["studio_annotations"] = studio_n
        c = entry["app_vs_ultralytics"]
        entry["pass"] = bool(counts_ok(len(ap_), len(ul)) and c["classes_a"] == c["classes_b"] and (c["max_abs_conf_diff"] or 0) < 0.05)
        report[f] = entry
        rows.append((f, entry["true_spots"], len(ul), len(onx), len(ap_), studio_n, c["matched"], c["mean_mask_iou"], c["mean_abs_conf_diff"], c["max_abs_conf_diff"], entry["pass"]))

    print(f"\nParity at imgsz={S}, conf={conf}, iou={iou}  (Ultralytics {ultralytics.__version__}, {pt.name} vs model/{cfg['model_file']})")
    hdr = ("image", "true", "ultra", "onnx-py", "app", "studio", "matched", "mask IoU", "mean dconf", "max dconf", "pass")
    fmt = "{:<16}{:>6}{:>7}{:>9}{:>6}{:>8}{:>9}{:>10}{:>12}{:>11}{:>6}"
    print(fmt.format(*hdr))
    for row in rows:
        print(fmt.format(*["-" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v)) for v in row]))
    print("\ncolumns: true = labelled spots; ultra / onnx-py / app / studio = spots found by each pipeline;")
    print("matched, mask IoU, dconf = app vs Ultralytics (same class, mask IoU >= 0.5).")

    summary = {
        "imgsz": S, "conf": conf, "iou": iou, "pt": pt.name, "model_sha256": cfg.get("model_sha256"),
        "ultralytics_version": ultralytics.__version__, "app_user_agent": app.get("userAgent"),
        "app_model_info": app.get("model"), "all_pass": all(e["pass"] for e in report.values()), "images": report,
    }
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n{'PASS' if summary['all_pass'] else 'FAIL'}: wrote {os.path.relpath(out_path, HERE)}")
    sys.exit(0 if summary["all_pass"] else 1)


if __name__ == "__main__":
    main()
