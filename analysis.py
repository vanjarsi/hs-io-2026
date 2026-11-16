"""Turning raw metrics into the numbers the judges score.

The ICICLE rubric (Model Performance, 30 pts) scores Precision, Recall and
F1 at 10 points each:  >=0.80 -> 10, 0.60-0.79 -> 8, 0.40-0.59 -> 5,
0.01-0.39 -> 2, 0 -> 0.

YOLO segmentation reports every metric twice: for boxes (B) and for masks (M).
Our task is segmentation, so the mask numbers are the ones that matter most.
"""
import re

METRICS = ["precision", "recall", "mAP50", "mAP50-95"]


def _kind(key):
    """'mask', 'box' or None, from a metric name like 'metrics/recall(M)'."""
    k = key.lower()
    if "(m)" in k or "mask" in k or re.search(r"[_/-]m$", k):
        return "mask"
    if "(b)" in k or "box" in k or re.search(r"[_/-]b$", k):
        return "box"
    return None


def _metric(key):
    """Which of METRICS a key holds (or None). Checks mAP50-95 before mAP50."""
    k = key.lower().replace("_", "-")
    if "map50-95" in k or "map-50-95" in k:
        return "mAP50-95"
    if "map50" in k or "map-50" in k:
        return "mAP50"
    if "precision" in k:
        return "precision"
    if "recall" in k:
        return "recall"
    return None


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def summarize(flat_metrics):
    """{'mask': {'precision':..,'recall':..,'F1':..,'mAP50':..,'mAP50-95':..}, 'box': {...}}"""
    out = {"mask": {}, "box": {}}
    for key, value in flat_metrics.items():
        kind, metric = _kind(key), _metric(key)
        if kind and metric and metric not in out[kind]:
            out[kind][metric] = float(value)
    for scores in out.values():
        if "precision" in scores and "recall" in scores:
            scores["F1"] = f1(scores["precision"], scores["recall"])
    return out


def rubric_points(value):
    """Points out of 10 for one of precision / recall / F1."""
    if value is None:
        return None
    if value >= 0.80:
        return 10
    if value >= 0.60:
        return 8
    if value >= 0.40:
        return 5
    if value > 0:
        return 2
    return 0


def performance_score(scores):
    """Model Performance points (out of 30) from one set of scores."""
    pts = [rubric_points(scores.get(m)) for m in ("precision", "recall", "F1")]
    return None if None in pts else sum(pts)


def next_band_gap(value):
    """How far a value is from the next rubric band (None if already top)."""
    for cutoff in (0.40, 0.60, 0.80):
        if value < cutoff:
            return cutoff - value
    return None


# ---------- Training-curve diagnostics (results.csv) ----------
# Judges asked: did the losses actually go down, did training "stick"
# (plateau / converge), did it overfit, and was 20 epochs enough?

LOSS_PARTS = ["box", "seg", "cls", "dfl"]


def loss_columns(curves, split):
    """{'box': 'train/box_loss', ...} for split 'train' or 'val'."""
    return {p: f"{split}/{p}_loss" for p in LOSS_PARTS if f"{split}/{p}_loss" in curves}


def total_loss(curves, split):
    """Sum of all loss parts per epoch, or None if the csv has none."""
    cols = list(loss_columns(curves, split).values())
    return curves[cols].sum(axis=1) if cols else None


def _rel_change(series, window):
    """Relative change from `window` epochs ago to the last epoch (+ = went up)."""
    s = series.dropna()
    if len(s) <= window or s.iloc[-1 - window] == 0:
        return None
    return (s.iloc[-1] - s.iloc[-1 - window]) / abs(s.iloc[-1 - window])


def convergence(curves, planned_epochs=20, window=5):
    """Plain-English diagnosis of one run's curves.

    Returns a dict of facts plus a list of findings (level, message), where
    level is 'good', 'warn' or 'info'.
    """
    epochs = curves["epoch"] if "epoch" in curves else curves.index + 1
    n = len(curves)
    facts = {"epochs_run": n, "planned_epochs": planned_epochs}
    findings = []

    train, val = total_loss(curves, "train"), total_loss(curves, "val")
    if train is not None and train.notna().sum() > 1:
        t = train.dropna()
        facts["train_loss_drop"] = (t.iloc[0] - t.iloc[-1]) / t.iloc[0]
    if val is not None and val.notna().sum() > 1:
        v = val.dropna()
        facts["val_loss_drop"] = (v.iloc[0] - v.iloc[-1]) / v.iloc[0]
        best_i = v.idxmin()
        facts["best_val_loss_epoch"] = int(epochs[best_i])

    map_col = "metrics/mAP50-95(M)"
    if map_col in curves and curves[map_col].notna().any():
        m = curves[map_col]
        best_i = m.idxmax()
        facts["best_map_epoch"] = int(epochs[best_i])
        facts["best_map"] = float(m[best_i])
        facts["map_gain_last_window"] = (
            float(m.dropna().iloc[-1] - m.dropna().iloc[-1 - window])
            if m.notna().sum() > window else None)

    # 1. Did learning happen at all?
    drop = facts.get("val_loss_drop")
    if drop is not None:
        if drop > 0.10:
            findings.append(("good", f"Validation loss fell {drop:.0%} from epoch 1 to the end: "
                                     "the model learned and the result generalises to unseen images."))
        else:
            findings.append(("warn", f"Validation loss only fell {drop:.0%}: the model barely "
                                     "improved on unseen images."))

    # 3. Overfitting: train still dropping while val goes up
    tr, vr = (_rel_change(train, window) if train is not None else None,
              _rel_change(val, window) if val is not None else None)
    overfit = tr is not None and vr is not None and tr < -0.02 and vr > 0.02
    if overfit:
        findings.append(("warn", f"Over the last {window} epochs training loss kept falling "
                                 f"({tr:.0%}) but validation loss rose ({vr:+.0%}): overfitting. "
                                 "Use the best.pt checkpoint, not last.pt."))

    # 4. Plateau ("stuck") vs still improving
    gain = facts.get("map_gain_last_window")
    if gain is not None:
        if gain > 0.02:
            findings.append(("info", f"Mask mAP50-95 was still rising (+{gain:.3f} over the last "
                                     f"{window} epochs): not converged yet, more epochs would "
                                     "likely help."))
        elif gain > 0.01:
            findings.append(("info", f"Mask mAP50-95 gains were slowing (+{gain:.3f} over the last "
                                     f"{window} epochs): close to converged."))
        else:
            msg = (f"Mask mAP50-95 was flat over the last {window} epochs "
                   f"({gain:+.3f}): training converged.")
            if vr is not None and vr < -0.03:
                msg += (f" Validation loss was still edging down ({vr:.0%}), so a few more "
                        "epochs might add a little.")
            findings.append(("info" if overfit else "good",
                             msg.replace("training converged", "the scores had levelled off")
                             if overfit else msg))

    # 2. Stopped before the planned epochs: patience vs time limit
    if n < planned_epochs:
        if gain is not None and gain > 0.02:
            findings.append(("warn", f"Stopped after {n} of {planned_epochs} epochs while still "
                                     "improving: probably hit the job's time limit, not early "
                                     "stopping. Check the end of the training log."))
        else:
            findings.append(("info", f"Stopped after {n} of {planned_epochs} epochs: early "
                                     "stopping (patience) because validation scores stopped "
                                     "improving."))
    if "best_map_epoch" in facts and facts["best_map_epoch"] < int(epochs.iloc[-1]) - window:
        findings.append(("info", f"Best mask mAP50-95 came at epoch {facts['best_map_epoch']}, "
                                 "well before the end."))

    facts["findings"] = findings
    return facts
