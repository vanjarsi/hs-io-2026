#!/usr/bin/env bash
# Export a trained model for the phone app. One command swaps the model:
#
#   ./export_model.sh <best.pt | model.onnx> <imgsz> <conf> ["run label"]
#
#   ./export_model.sh models/runB_best.pt 640 0.30 "Workflow Studio run #31"
#   ./export_model.sh final_best.pt 800 0.30 "Workflow Studio run #NN"
#   ./export_model.sh final.onnx 800 0.30          # ONNX already exported: config-only
#   PRECISION=auto ./export_model.sh best.pt 640 0.30   # try FP16, keep it only if it matches FP32
#
# It replaces model/model.onnx and model/config.json. Nothing else changes.
# Needs Python with: ultralytics onnx onnxruntime onnxslim onnxconverter-common
set -euo pipefail
cd "$(dirname "$0")"

if [ $# -lt 3 ]; then
  sed -n '2,12p' "$0"; exit 1
fi

SRC="$1"
case "$SRC" in /*) ;; *) SRC="$OLDPWD/$SRC" ;; esac
[ -f "$SRC" ] || SRC="$1"   # also accept a path relative to phone_app/

if [ -n "${PHONE_APP_PYTHON:-}" ]; then PY="$PHONE_APP_PYTHON"
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else PY=python3; fi

"$PY" export_model.py "$SRC" "${@:2}"
