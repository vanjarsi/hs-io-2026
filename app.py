"""Apple leaf disease segmentation: results dashboard.

Run with:  streamlit run app.py
Drop each training run's metrics.json in via the sidebar (or put it in
data/runs/<run-name>/) and the page scores it against the ICICLE rubric.
"""
import plotly.graph_objects as go
import streamlit as st

import analysis
import data

st.set_page_config(page_title="Leaf Disease Results", layout="wide")

# Categorical palette, fixed order. Color follows the run, never its rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TARGET = 0.80  # rubric: >= 0.80 earns full points

# ---------- Sidebar: add a run ----------
with st.sidebar:
    st.header("Add a run")
    st.caption("Download metrics.json (and results.csv) from the run's archive "
               "folder in the Workflow Studio, then upload them here.")
    with st.form("add_run", clear_on_submit=True):
        name = st.text_input("Run name", placeholder="e.g. 01-baseline-640")
        notes = st.text_area("What changed in this run?",
                             placeholder="e.g. no hue shift, imgsz 640, 20 epochs")
        metrics_file = st.file_uploader("metrics.json", type=["json"])
        csv_file = st.file_uploader("results.csv (optional)", type=["csv"])
        if st.form_submit_button("Save run"):
            if not name or not metrics_file:
                st.error("Need a run name and a metrics.json.")
            else:
                try:
                    data.save_run(name.strip(), metrics_file.getvalue(),
                                  csv_file.getvalue() if csv_file else None, notes)
                    st.success(f"Saved {name}")
                except ValueError:
                    st.error("That metrics.json isn't valid JSON.")

st.title("Apple Leaf Disease Segmentation: Results")
st.caption("YOLO11 instance segmentation of apple scab, black rot and cedar apple rust, "
           "trained in the ICICLE No-code Workflow Studio on AWS.")

runs = data.list_runs()
if not runs:
    st.info("No runs yet. When a training run finishes, upload its metrics.json "
            "in the sidebar.")
    st.stop()

kind = st.radio("Score using", ["mask", "box"], horizontal=True,
                help="Our task is segmentation, so mask (M) metrics are the ones that count.")

summaries = {r: analysis.summarize(data.load_metrics(r))[kind] for r in runs}
colors = {r: SERIES[i % len(SERIES)] for i, r in enumerate(runs)}

# ---------- Headline: best run ----------
scored = {r: analysis.performance_score(s) for r, s in summaries.items()}
valid = {r: v for r, v in scored.items() if v is not None}
if valid:
    best = max(valid, key=valid.get)
    s = summaries[best]
    st.subheader(f"Best run: {best}")
    cols = st.columns(5)
    for col, m in zip(cols, ["precision", "recall", "F1", "mAP50", "mAP50-95"]):
        v = s.get(m)
        col.metric(m, "n/a" if v is None else f"{v:.3f}")
    st.markdown(f"**Model Performance: {valid[best]} / 30 rubric points**")
else:
    st.warning(f"Couldn't find {kind} precision/recall in the uploaded metrics.json. "
               "Check the raw keys below.")

# ---------- Comparison chart ----------
st.subheader("Precision, recall and F1 by run")
fig = go.Figure()
labels = ["precision", "recall", "F1"]
for r in runs:
    vals = [summaries[r].get(m) for m in labels]
    fig.add_bar(name=r, x=labels, y=vals, marker_color=colors[r],
                marker_line_width=2, marker_line_color="rgba(0,0,0,0)",
                hovertemplate=f"<b>{r}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>")
fig.add_hline(y=TARGET, line_dash="dash", line_width=1, line_color="gray",
              annotation_text="0.80 = full rubric points", annotation_position="top left")
fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08,
                  yaxis=dict(range=[0, 1], title=None, gridcolor="rgba(128,128,128,0.15)"),
                  xaxis_title=None, legend=dict(orientation="h", y=1.12),
                  margin=dict(t=40, b=10, l=10, r=10), height=380)
st.plotly_chart(fig, use_container_width=True)

# ---------- Table view (same numbers, plus rubric points) ----------
rows = []
for r in runs:
    s = summaries[r]
    row = {"run": r}
    for m in ["precision", "recall", "F1", "mAP50", "mAP50-95"]:
        row[m] = None if s.get(m) is None else round(s[m], 3)
    row["rubric pts (/30)"] = scored[r]
    row["notes"] = data.load_notes(r)
    rows.append(row)
st.dataframe(rows, use_container_width=True, hide_index=True)

# ---------- What would earn more points ----------
if valid:
    tips = []
    for m in labels:
        v = summaries[best].get(m)
        gap = analysis.next_band_gap(v) if v is not None else None
        if gap is not None:
            tips.append(f"- **{m}** is {v:.2f}: {gap:.2f} more reaches the next rubric band.")
    if tips:
        st.markdown("**Closest wins for the best run:**\n" + "\n".join(tips))
        st.caption("Low precision means too many false spots: try a higher confidence threshold. "
                   "Low recall means missed spots: try a larger image size or more epochs.")

# ---------- Training curves: did the losses go down, did it converge? ----------
LAYOUT = dict(hovermode="x unified", height=320, margin=dict(t=40, b=10, l=10, r=10),
              legend=dict(orientation="h", y=-0.25), xaxis_title="epoch")
ICON = {"good": "✅", "warn": "⚠️", "info": "ℹ️"}

st.subheader("Training curves")
pick = st.selectbox("Run", runs, index=len(runs) - 1)
curves = data.load_curves(pick)
if curves is None:
    st.caption("Upload this run's results.csv to see its loss and epoch curves.")
else:
    epoch = curves["epoch"] if "epoch" in curves else curves.index + 1
    diag = analysis.convergence(curves)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Epochs run", f"{diag['epochs_run']} / {diag['planned_epochs']}")
    c2.metric("Val loss drop", "n/a" if "val_loss_drop" not in diag else f"{diag['val_loss_drop']:.0%}")
    c3.metric("Best mask mAP50-95", "n/a" if "best_map" not in diag else f"{diag['best_map']:.3f}")
    c4.metric("Best epoch", diag.get("best_map_epoch", "n/a"))
    for level, msg in diag["findings"]:
        st.markdown(f"{ICON[level]} {msg}")

    left, right = st.columns(2)
    with left:
        part = st.radio("Loss", ["total"] + analysis.LOSS_PARTS, horizontal=True,
                        help="seg = mask outline, box = box position, cls = disease class, "
                             "dfl = box edge precision. total = all four added up.")
        f = go.Figure()
        for i, split in enumerate(["train", "val"]):
            if part == "total":
                y = analysis.total_loss(curves, split)
            else:
                col = analysis.loss_columns(curves, split).get(part)
                y = curves[col] if col else None
            if y is not None:
                f.add_scatter(x=epoch, y=y, name=split, mode="lines+markers",
                              line=dict(width=2, color=SERIES[i]), marker=dict(size=6))
        f.update_layout(title=f"{part} loss: train vs validation (lower is better)", **LAYOUT)
        st.plotly_chart(f, use_container_width=True)
        st.caption("Both lines falling together means real learning. Train falling while "
                   "validation rises means overfitting.")
    with right:
        f = go.Figure()
        for i, m in enumerate(["precision", "recall", "mAP50", "mAP50-95"]):
            col = f"metrics/{m}(M)"
            if col in curves:
                f.add_scatter(x=epoch, y=curves[col], name=m, mode="lines+markers",
                              line=dict(width=2, color=SERIES[i]), marker=dict(size=6))
        f.add_hline(y=TARGET, line_dash="dash", line_width=1, line_color="gray")
        f.update_layout(title="Mask metrics by epoch (higher is better)",
                        yaxis=dict(range=[0, 1]), **LAYOUT)
        st.plotly_chart(f, use_container_width=True)
        st.caption("A line that flattens out means that metric has converged. "
                   "Dashed line = 0.80 rubric target.")

# ---------- All runs on the same axes ----------
with_curves = [r for r in runs if data.load_curves(r) is not None]
if len(with_curves) > 1:
    st.subheader("Runs compared epoch by epoch")
    left, right = st.columns(2)
    for box, (title, getter) in zip(
            (left, right),
            [("Validation total loss", lambda c: analysis.total_loss(c, "val")),
             ("Mask mAP50-95", lambda c: c.get("metrics/mAP50-95(M)"))]):
        f = go.Figure()
        for r in with_curves:
            c = data.load_curves(r)
            y = getter(c)
            if y is not None:
                f.add_scatter(x=c["epoch"] if "epoch" in c else c.index + 1, y=y, name=r,
                              mode="lines+markers", line=dict(width=2, color=colors[r]),
                              marker=dict(size=6))
        f.update_layout(title=title, **LAYOUT)
        box.plotly_chart(f, use_container_width=True)

with st.expander("Raw metrics.json keys"):
    st.json(data.load_metrics(pick))
