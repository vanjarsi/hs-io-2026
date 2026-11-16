"""Export a trained YOLO-seg model for the phone app.

    python export_model.py <best.pt | model.onnx> <imgsz> <conf> [run label]

Writes model/model.onnx and model/config.json. The web app reads every model
setting from config.json, so swapping models is this one command.

Options (environment variables):
    PRECISION=auto|fp32|fp16   default auto: convert to FP16 (half the file size)
                               and keep it only if its detections match FP32
                               on the bundled samples; FP32 is then kept next
                               to it as model_fp32.onnx, the app's fallback.
    NMS_IOU=0.7                NMS IoU written to the config.
    TRAIN_PX=256               pixel size of the training images (used by the
                               app's "match training resolution" option).
    OPSET=17                   ONNX opset.
"""
import datetime
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_DIR = Path(__import__("os").environ.get("PHONE_APP_OUT", str(HERE / "model")))  # override to export a second model side by side


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def class_entries(names):
    disp = json.loads((HERE / "class_display.json").read_text())
    palette, known, out, used = disp["palette"], disp["classes"], [], set()
    for i in sorted(names):
        k = known.get(names[i], {})
        out.append({"id": int(i), "name": names[i], "label": k.get("label", names[i].replace("_", " ")), "color": k.get("color")})
        used.add(k.get("color"))
    free = [c for c in palette if c not in used] or palette
    n = 0
    for e in out:
        if not e["color"]:
            e["color"] = free[n % len(free)]
            n += 1
    return out


def sample_images():
    import cv2

    imgs = []
    for p in sorted((HERE / "samples").glob("*.jpg")):
        im = cv2.imread(str(p))
        if im is not None:
            imgs.append((p.name, cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
    if not imgs:  # no samples bundled yet: structured noise still exercises the graph
        rng = np.random.default_rng(0)
        imgs = [("noise", rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))]
    return imgs


def fp16_matches(fp32_path, fp16_path, imgsz, conf, iou):
    """True if FP16 gives the same detections as FP32 on the bundled samples (onnxruntime, CPU)."""
    import onnxruntime as ort

    from onnx_ref import run_onnx

    a = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(fp16_path), providers=["CPUExecutionProvider"])
    ok = True
    for name, img in sample_images():
        ra = run_onnx(a, img, imgsz, conf, iou, with_masks=False)
        rb = run_onnx(b, img, imgsz, conf, iou, with_masks=False)
        same_n = len(ra["score"]) == len(rb["score"])
        same_c = same_n and bool((ra["cls"] == rb["cls"]).all())
        dmax = float(np.abs(ra["score"] - rb["score"]).max()) if same_n and len(ra["score"]) else 0.0
        print(f"  fp16 check {name}: fp32={len(ra['score'])} fp16={len(rb['score'])} classes_equal={same_c} max_conf_diff={dmax:.4f}")
        ok &= same_n and same_c and dmax < 0.02
    return ok


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    src, imgsz, conf = Path(sys.argv[1]).resolve(), int(sys.argv[2]), float(sys.argv[3])
    run_label = sys.argv[4] if len(sys.argv) > 4 else None
    precision_req = os.environ.get("PRECISION", "auto").lower()
    iou = float(os.environ.get("NMS_IOU", "0.7"))
    train_px = int(os.environ.get("TRAIN_PX", "256"))
    opset = int(os.environ.get("OPSET", "17"))
    if not src.exists():
        sys.exit(f"not found: {src}")
    OUT_DIR.mkdir(exist_ok=True)
    tmp = OUT_DIR / "model.tmp.onnx"
    meta = {"params": None, "gflops": None, "model_name": None, "ultralytics_version": None, "train_args": None}

    import onnx

    if src.suffix.lower() == ".onnx":
        # Config-only path: the ONNX file already exists, imgsz/conf come from the command line.
        print(f"Using existing ONNX file {src.name} (config-only path)")
        shutil.copyfile(src, tmp)
        props = {p.key: p.value for p in onnx.load(str(tmp), load_external_data=False).metadata_props}
        names = None
        if "names" in props:
            import ast

            names = {int(k): v for k, v in ast.literal_eval(props["names"]).items()}
        elif (OUT_DIR / "config.json").exists():
            old = json.loads((OUT_DIR / "config.json").read_text())
            names = {c["id"]: c["name"] for c in old["classes"]}
            print("  class names not in the ONNX metadata: reusing those of the previous config.json")
        else:
            sys.exit("class names unknown: the ONNX file has no 'names' metadata and there is no previous config.json")
        meta["ultralytics_version"] = props.get("version")
        meta["model_name"] = props.get("description", "").split(" model")[0].replace("Ultralytics ", "") or None
    else:
        import ultralytics
        from ultralytics import YOLO

        model = YOLO(str(src))
        names = dict(model.names)
        try:
            from ultralytics.utils.torch_utils import get_flops, get_num_params

            import copy

            fused = copy.deepcopy(model.model).fuse(verbose=False)  # the form that is exported, as in the "summary (fused)" log line
            meta["params"] = int(get_num_params(fused))
            meta["gflops"] = round(float(get_flops(fused, imgsz)), 1)
        except Exception as e:  # informational only
            print("  could not read params/GFLOPs:", e)
        targs = dict(getattr(model, "ckpt", None) and model.ckpt.get("train_args") or {})
        base = Path(str(targs.get("model", src.stem))).stem
        meta["model_name"] = "YOLO" + base[4:] if base.lower().startswith("yolo") else base
        meta["train_args"] = {k: targs.get(k) for k in ("epochs", "batch", "imgsz", "seed") if k in targs}
        meta["ultralytics_version"] = ultralytics.__version__
        exported = model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=True, dynamic=False, nms=False, device="cpu")
        shutil.move(str(exported), tmp)

    precision = "fp32"
    fallback = None
    for stale in ("model_fp32.onnx", "model_fp16.onnx"):
        (OUT_DIR / stale).unlink(missing_ok=True)
    if precision_req in ("fp16", "auto"):
        tmp16 = OUT_DIR / "model.tmp16.onnx"
        try:
            import warnings

            from onnxconverter_common import float16

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # "value will be truncated" notes for near-zero weights
                m16 = float16.convert_float_to_float16(onnx.load(str(tmp)), keep_io_types=True)
            # The converter leaves float16 type notes on tensors it keeps in float32 (around Resize),
            # which stops the file loading. They are optional, so drop them.
            del m16.graph.value_info[:]
            onnx.save(m16, str(tmp16))
            good = fp16_matches(tmp, tmp16, imgsz, conf, iou)
            print(f"  FP16: {tmp16.stat().st_size/1e6:.2f} MB vs FP32 {tmp.stat().st_size/1e6:.2f} MB, detections match FP32: {good}")
            if good or precision_req == "fp16":
                if not good:
                    print("  WARNING: FP16 kept on request although detections differ from FP32")
                fallback = "model_fp32.onnx"  # the app falls back to it if FP16 will not start on a device
                shutil.copyfile(tmp, OUT_DIR / fallback)
                tmp16.replace(tmp)
                precision = "fp16"
        except Exception as e:
            print("  FP16 conversion failed, shipping FP32:", str(e)[:300])
        finally:
            tmp16.unlink(missing_ok=True)

    # Read the real input/output names and shapes back from the file.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    if "float16" in inp.type or any(not isinstance(d, int) for d in inp.shape):
        tmp.unlink(missing_ok=True)
        sys.exit(f"the ONNX input is {inp.type} {inp.shape}: the app needs a float32 input with a fixed shape "
                 "(export with half=False, dynamic=False; this script converts the weights to FP16 itself)")
    outs = sess.run(None, {inp.name: np.zeros(inp.shape, dtype=np.float32)})
    outputs = [{"name": o.name, "shape": list(a.shape)} for o, a in zip(sess.get_outputs(), outs)]
    if list(inp.shape)[2:] != [imgsz, imgsz]:
        sys.exit(f"the ONNX input is {inp.shape}, which does not match imgsz {imgsz}")
    proto = next(o for o in outputs if len(o["shape"]) == 4)
    det = next(o for o in outputs if len(o["shape"]) == 3)
    nc = min(det["shape"][1:]) - 4 - proto["shape"][1]
    if nc != len(names):
        sys.exit(f"the model outputs {nc} class scores but {len(names)} class names are known")

    final = OUT_DIR / "model.onnx"
    tmp.replace(final)
    try:
        source_rel = str(src.relative_to(HERE))
    except ValueError:
        source_rel = src.name
    config = {
        "model_file": "model.onnx",
        "model_bytes": final.stat().st_size,
        "model_sha256": sha256(final),
        "model_name": meta["model_name"],
        "task": "segment",
        "precision": precision,
        "fallback_model_file": fallback,
        "fallback_model_bytes": (OUT_DIR / fallback).stat().st_size if fallback else None,
        "fallback_precision": "fp32" if fallback else None,
        "classes": class_entries(names),
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "max_det": 300,
        "mask_threshold": 0.5,
        "pad_value": 114,
        "input": {"name": inp.name, "shape": list(inp.shape), "dtype": "float32", "layout": "NCHW", "color": "RGB", "scale": "1/255"},
        "outputs": outputs,
        "phone_photo": {"match_training_resolution": True, "center_crop_square": True, "train_image_px": train_px},
        "source_model": source_rel,
        "source_sha256": sha256(src),
        "training_run": run_label,
        "train_args": meta["train_args"],
        "params": meta["params"],
        "gflops": meta["gflops"],
        "opset": opset if src.suffix.lower() != ".onnx" else None,
        "ultralytics_version": meta["ultralytics_version"],
        "export_date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"\nWrote {final.relative_to(HERE)} ({final.stat().st_size/1e6:.2f} MB, {precision}) and model/config.json")
    print(f"  classes: {[c['name'] for c in config['classes']]}  imgsz={imgsz} conf={conf} iou={iou}")
    print(f"  input {inp.name} {list(inp.shape)}  outputs {[(o['name'], o['shape']) for o in outputs]}")


if __name__ == "__main__":
    main()
