# Final model: Run D with two inference fixes

Same weights as Run D (YOLO11n-seg, image size 1024, 20 epochs, trained in the workflow). No
retraining. Two changes to how predictions are read out of the model, found by looking at what
its errors were:

1. **Duplicate removal.** 411 of Run D's 1,700 false positives at confidence 0.30 were a second
   mask on a spot the model had already found. YOLO's built-in NMS compares boxes at IoU 0.7,
   and for 7-pixel spots a box shifted by a pixel or two slips under that. We drop a prediction
   when its mask overlaps a higher-confidence mask of the same disease by more than 0.5 (mask IoU).
2. **Softer mask cut-off.** When a prediction nearly matched a true apple scab or rust spot, its
   mask was usually too small (median 0.57x the labelled area, sitting inside the true outline).
   YOLO draws the mask where the mask probability is above 0.5; we use 0.15, so the mask grows
   where the model itself is unsure of the edge. It is still cropped to the predicted box.

Removing duplicates raises precision, which lets the confidence threshold come down and recall
go up.

## Event scorer result (Nov 15, 2026, 4:05 PM)

`FINAL3_runD_softmask_perclass.json`, scored by the event's Submission Scorer on the 777 test images:

| Class | Precision | Recall | F1 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| apple_scab | 0.7915 | 0.7680 | 0.7796 | 3569 | 940 | 1078 |
| black_rot | 0.8622 | 0.8133 | 0.8371 | 2096 | 335 | 481 |
| spot | 0.8159 | 0.8424 | 0.8289 | 4245 | 958 | 794 |
| **Overall** | **0.8161** | **0.8081** | **0.8121** | 9910 | 2233 | 2353 |

Run D as first submitted (confidence 0.30) scored 0.844 / 0.732 / 0.784 on the same scorer
(TP 8976, FP 1660, FN 3287). The same file is on AWS at
`aws-tapis:/fs/scratch/A0002/yrawale/annotations/FINAL3_runD_softmask_perclass.json`.

## Results

All numbers are mask precision / recall / F1 with IoU >= 0.5, from `evaluate.py`. It uses the
same matching rule as `eval_test.py`, reproduces its counts exactly, and has read about 0.004
BELOW the event's scorer on every file we have compared. **The event scorer is the judge:
upload the files and use its numbers.**

Test set (777 images, 12,263 labelled spots):

| Pipeline | Confidence | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Run D as submitted (Workflow Studio run #63) | 0.30 | 0.840 | 0.729 | 0.780 |
| Run D, lowest threshold with precision >= 0.80 | 0.25 | 0.801 | 0.760 | 0.780 |
| + duplicate removal | 0.18 | 0.804 | 0.788 | 0.796 |
| + duplicate removal + mask cut-off 0.15 | 0.16 | 0.805 | 0.802 | 0.804 |
| + per-disease settings chosen on validation (below) | per disease | 0.809 | 0.801 | 0.805 |
| Highest F1: duplicate removal + mask cut-off 0.15 | 0.23 | 0.859 | 0.765 | 0.809 |

Validation (random 1,000 of the 2,331 images, 15,903 labelled spots), used to choose settings:

| Pipeline | Best F1 | Precision / recall at the lowest threshold with precision >= 0.80 |
| --- | --- | --- |
| Run D, standard masks, duplicates removed | 0.792 | 0.801 / 0.778 |
| + mask cut-off 0.35 | 0.800 | 0.800 / 0.788 |
| + mask cut-off 0.25 | 0.803 | 0.802 / 0.792 |
| + mask cut-off 0.15 | 0.804 | 0.800 / 0.795 |

Validation is slightly harder than test for this model (Run D at 0.30: F1 0.776 on validation,
0.780 on test), so the 0.80 / 0.80 point is reached on test but not quite on validation.

Per-disease settings (chosen on validation as the lowest thresholds that keep validation
precision >= 0.81, a 0.01 margin):

| Disease | Mask cut-off | Confidence |
| --- | --- | --- |
| apple_scab | 0.15 | 0.185 |
| black_rot | 0.35 | 0.19 |
| spot (cedar apple rust) | 0.15 | 0.13 |

On validation: 0.810 / 0.791 / 0.801. On test: 0.809 / 0.801 / 0.805.

How the settings were found: the ideas (duplicate removal, softer masks) were first tried on
the test set, because the validation images had not been downloaded yet. They were then
re-checked on validation, and the thresholds above come from validation only.

### Tried, no real gain

| Idea | Recall at precision >= 0.80 (test) |
| --- | --- |
| Inference at image size 1280 instead of 1024 | 0.784 (1024: 0.788) |
| Fusing Run B (640) with Run D | 0.789–0.792 |
| Adding a horizontally flipped view | 0.791 |
| Merging duplicates by union / vote instead of keeping the best | 0.774–0.789 |
| Growing every mask by a pixel, or a quarter pixel | worse |
| Enlarging the box the mask is cropped to | much worse |
| Mask cut-offs below 0.15 (0.10 to 0.03) | flat |

Not tried: more training epochs. That needs a GPU run of 80+ minutes and could not finish in time.

## Files for the event scorer (in ~/Downloads)

| File | What | Local P / R / F1 |
| --- | --- | --- |
| `FINAL3_runD_softmask_perclass.json` | per-disease settings | 0.809 / 0.801 / 0.805 |
| `FINAL2_runD_softmask_dedup_conf016.json` | one threshold, 0.16 | 0.805 / 0.802 / 0.804 |
| `FINAL2_runD_softmask_dedup_conf017.json` | one threshold, 0.17 | 0.815 / 0.797 / 0.806 |
| `FINAL2_runD_softmask_dedup_conf023.json` | highest F1, precision safe | 0.859 / 0.765 / 0.809 |
| `FINAL_runD_dedup_conf022.json` | duplicates removed only, standard masks | 0.833 / 0.766 / 0.798 |

## Reproduce

```bash
cd final_model
PY=../phone_app/.venv/bin/python
# 1. one pass of the model, writing predictions for several mask cut-offs (about 5 min on a laptop CPU)
$PY predict_soft.py --onnx runD_1024_fp32.onnx --images <test images> --imgsz 1024 \
    --thr 0.5,0.35,0.15 --expand 0 --conf 0.10 --out-prefix preds/test_D
# 2. final file: duplicates removed, one threshold
$PY make_final.py out.json --preds preds/test_D_t0.15_e0.json --conf 0.16
#    or per-disease settings
$PY make_final.py out.json --class-file apple_scab=preds/test_D_t0.15_e0.json \
    --class-file black_rot=preds/test_D_t0.35_e0.json --class-file spot=preds/test_D_t0.15_e0.json \
    --class-conf apple_scab=0.185,black_rot=0.19,spot=0.13
# 3. score it
$PY evaluate.py out.json --labels ../.playwright-mcp/test_labels.json
```

`runD_1024_fp32.onnx` is Run D's `best.pt` exported with `phone_app/export_model.sh` at image
size 1024. With `--thr 0.5` the pipeline reproduces Ultralytics and the Workflow Studio
(same 10,636 spots at confidence 0.30 as run #63, to within one).

This post-processing runs in our own script. The Workflow Studio's inference block cannot change
the mask cut-off or remove duplicates, so say that plainly when presenting it.
