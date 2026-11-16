# Leaf Check: the trained model, running on the phone

> The app runs the exact model our workflow trained. Training happens in the
> Workflow Studio on a GPU cluster; the farmer runs the model in their pocket.

A single static web page that runs our YOLO11n-seg apple-leaf disease model
**inside the phone's browser** with onnxruntime-web (WebAssembly). No app
store, no server doing the inference: the photo never leaves the phone.

- Input: a camera photo, a photo from the gallery, or one of 6 bundled test leaves.
- Output: a colour-coded mask and outline per disease spot, counts per disease,
  a list of detections with confidence, a clear "Healthy" state, and the model
  size, load time and per-stage timings in milliseconds.

## Deployed model: Run B

| | |
| --- | --- |
| Model | YOLO11n-seg, Workflow Studio run #31 ("Run B"), 20 epochs at imgsz 640 |
| Classes | 0 `apple_scab` (Apple scab), 1 `black_rot` (Black rot), 2 `spot` (Cedar apple rust) |
| Size | 2,849,953 parameters, 9.7 GFLOPs (fused). `best.pt` 6.0 MB |
| Shipped file | `model/model.onnx`, **5.86 MB** (FP16 weights, float32 inputs/outputs). FP32 fallback: 11.69 MB |
| Settings | imgsz 640, confidence 0.30 (best test F1: 0.768), NMS IoU 0.7 |
| Export | 2026-11-15, Ultralytics 8.4.184, ONNX opset 17, laptop CPU, about 10 s |

Test-set scores for this model (777 images, scored by `eval_test.py` on the
Workflow Studio's predictions, not by this app): precision 0.830, recall
0.714, F1 0.768 at confidence 0.30.

## Measured results

### Parity with Ultralytics (2026-11-15)

`parity_check.py` on the 6 bundled test leaves, imgsz 640, conf 0.30, IoU 0.7.
The app ran the shipped FP16 file in headless Chrome 158.

| Image | Labelled spots | Ultralytics `.pt` | ONNX in Python | **App (browser)** | Studio annotations | App spots matched to Ultralytics | Mean mask IoU | Max confidence difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `healthy_1.jpg` | 0 | 0 | 0 | **0** | 0 | – | – | – |
| `scab_1.jpg` | 11 | 11 | 11 | **11** | 11 | 11 / 11 | 0.998 | 0.005 |
| `black_rot_1.jpg` | 10 | 9 | 9 | **9** | 9 | 9 / 9 | 0.997 | 0.001 |
| `rust_1.jpg` | 15 | 15 | 15 | **15** | 15 | 15 / 15 | 0.985 | 0.008 |
| `scab_2.jpg` | 7 | 6 | 6 | **6** | 6 | 6 / 6 | 0.994 | 0.003 |
| `black_rot_2.jpg` | 4 | 5 | 5 | **5** | 5 | 5 / 5 | 0.974 | 0.005 |

- Same counts and same classes in all four pipelines on every image.
- Against the Workflow Studio's saved predictions for the same images
  (`runB_test_annotations.json`), every app detection also matched: mask IoU
  0.974–0.998, confidence within 0.008.
- "Labelled spots" is the ground truth; the model itself misses or adds a spot
  on three of these leaves, in the app exactly as on the server.
- This checks 6 images, not the whole test set.

### Size and speed

| | FP16 (shipped) | FP32 (fallback) |
| --- | --- | --- |
| File size | 5.86 MB | 11.69 MB |
| Detections vs FP32 on the 6 samples | identical counts and classes, confidence within 0.002 | – |

**Laptop** (Apple M1, 8 GB, macOS 26.4, headless Chrome 158, onnxruntime-web
1.20.1 WASM; median of 54 runs over the 6 samples):

| | Inference, 4 threads | Inference, 1 thread | Preprocess | Postprocess |
| --- | --- | --- | --- | --- |
| FP16 | 116 ms (best 106) | 296 ms | 8 ms | 0.4 ms |
| FP32 | 104 ms (best 93) | 283 ms | 8 ms | 0.3 ms |

FP16 costs about 5–10% in speed for half the download. An earlier run of the
same test while the laptop was busy gave 158 ms (FP16) and 151 ms (FP32) at 4
threads, so expect that much variation.

Model start-up on the laptop: about 0.7–0.9 s to create the session plus a
0.3–0.4 s warm-up run (local server, model already downloaded in under 0.1 s).
For comparison, the same model takes 11.3 ms per image on the NVIDIA A10G GPU
used for training.

**Larger input size** (what the final model may use): the same Run B weights
exported at imgsz 800 took about 210 ms per image on the busy laptop (4
threads), against about 158 ms at 640 in the same conditions. The file size barely changes (5.91 MB).

**Phone:** not measured yet. The app shows model size, load time and the
per-stage times on screen; fill this in after the phone test.

### "Match training resolution": on or off?

Tried on one real field photo (cedar apple rust on a crab-apple leaf,
3008×2000 px, Wikimedia Commons) and a close-up crop of it:

| Photo | Match on | Match off |
| --- | --- | --- |
| Whole photo, centre-cropped | 51 spots | 49 spots |
| Close-up of the leaf | 30 spots | 30 spots |
| Two photos that are not apple leaves (false alarms) | 6 and 2 | 11 and 10 |

On the leaf the two settings behave the same; on non-leaf photos "on" raises
fewer false alarms, so it stays **on** by default. This is three photos, not a
study.

What that photo also showed, and what to say if a judge asks: the model found
the lesions, but called most of them black rot instead of cedar apple rust and
also marked blurred background. It was trained only on single leaves on plain
paper. For a live demo, photograph a test leaf shown on a screen, or a leaf on
a plain background.

## Files

| File | What it is |
| --- | --- |
| `index.html`, `style.css`, `app.js` | The page and its UI. |
| `engine.js`, `worker.js` | Preprocessing, inference and decoding, run in a Web Worker so the UI never freezes. |
| `model/model.onnx`, `model/config.json` | The exported model and **all** its settings. Written by `export_model.sh`. |
| `model/model_fp32.onnx` | FP32 copy, used automatically if a device cannot start the FP16 file. |
| `samples/` | 6 test-set leaves and `samples.json` (labels, true spot counts). |
| `export_model.sh`, `export_model.py` | One-command export and model swap. |
| `class_display.json` | Display name and colour per class (Okabe-Ito colourblind-safe palette). |
| `serve.py` | Local static server (127.0.0.1 only, serves only the app files). |
| `parity_check.py`, `onnx_ref.py`, `tools/run_app.js` | Parity check against Ultralytics. |
| `models/` | Trained `.pt` files. Not served, not committed. |

## Set up (once)

Export needs Python 3.10–3.12 with Ultralytics. The repo's main `.venv` is
Python 3.15, which PyTorch does not support yet, so the app has its own:

```bash
cd phone_app
python3.12 -m venv .venv
.venv/bin/pip install ultralytics==8.4.184 onnx onnxruntime onnxslim onnxconverter-common
```

Laptop CPU is enough: an export takes about 10 seconds.

## Export / swap the model: one command

```bash
./export_model.sh <best.pt> <imgsz> <conf> ["run label"]

./export_model.sh models/runB_best.pt 640 0.30 "Workflow Studio run #31"   # what is deployed now
./export_model.sh ~/Downloads/final_best.pt 800 0.30 "Workflow Studio run #NN"  # the final model
```

This replaces `model/model.onnx` and `model/config.json`. Nothing else
changes: reload the page and the app picks up the new classes, image size,
thresholds and model facts. **Use the `imgsz` the model was trained with** and
the confidence threshold with the best F1 on the test set.

- **Only an ONNX file?** `./export_model.sh final.onnx 800 0.30` copies it and
  writes the config from the command line ("config-only" path). Class names are
  read from the ONNX metadata Ultralytics embeds, or reused from the previous
  `config.json`.
- **FP16.** By default the script also converts the weights to 16-bit floats,
  which halves the download. It keeps FP16 only if its detections match FP32
  on the bundled samples (same count, same classes, confidences within 0.02),
  and keeps the FP32 file next to it as the app's fallback. Force one with
  `PRECISION=fp32` or `PRECISION=fp16`.
- Other settings: `NMS_IOU=0.7`, `TRAIN_PX=256` (size of the training images),
  `OPSET=17`.
- After a swap, re-run the parity check (below).

What the export does: `YOLO(best.pt).export(format="onnx", imgsz, opset=17,
simplify=True, dynamic=False, nms=False)`, then the FP16 conversion with
`onnxconverter_common` (`keep_io_types=True`, so inputs and outputs stay
float32), then it reads the real input/output names and shapes back from the
file and writes them into `config.json` with the class names, colours,
thresholds, source model hash, training run, export date and Ultralytics version.

## Serve

```bash
python3 serve.py            # http://127.0.0.1:8000
```

`serve.py` listens on this computer only and serves only the page, `model/`
and `samples/`. It sends the two headers (COOP/COEP) that allow multi-threaded
WebAssembly, which makes inference 2–3x faster.

For a phone you need HTTPS in front of it:

| Option | What it exposes | Needs |
| --- | --- | --- |
| **Cloudflare quick tunnel** `cloudflared tunnel --url http://localhost:8000` | A random, unlisted `https://….trycloudflare.com` address. Anyone who has the address can load the page, the ONNX model and the 6 sample leaves while the tunnel runs. Nothing else on the laptop. Stops when you press Ctrl-C. | The `cloudflared` program. No account. |
| **Tailscale Serve** `tailscale serve --bg 8000` | Only devices signed in to your own Tailscale network. Not reachable from the internet. | Tailscale running on laptop and phone, same account, HTTPS enabled for the tailnet. |
| Same Wi-Fi, plain HTTP | Everyone on that Wi-Fi. | Works for photos, but without HTTPS the browser allows only one thread (about 3x slower), and campus Wi-Fi often blocks device-to-device traffic. |

## How it works

1. **Photo in.** EXIF rotation is applied (`createImageBitmap(file,
   {imageOrientation: "from-image"})`).
2. **Match training resolution** (on by default). The model only ever saw
   256×256 px close-up leaf crops, upscaled to `imgsz`. A 4000×3000 phone photo
   resized straight to `imgsz` shows spots at a very different scale. So the
   photo is centre-cropped to a square, shrunk until its long side is 256 px,
   and only then letterboxed up to `imgsz`, exactly as the training images
   were. Trade-off: it throws away camera detail, so a spot must cover at least
   a few pixels of a 256 px frame: fill the frame with one leaf. Both steps can
   be switched off under "Photo options". The bundled test leaves are already
   256×256, so the options do not change them.
3. **Preprocess like Ultralytics.** RGB, letterbox to `imgsz`×`imgsz` keeping
   the aspect ratio (bilinear, pad value 114), divide by 255, NCHW float32.
4. **Run** `model.onnx` with onnxruntime-web 1.20.1 (pinned), WASM backend,
   in a Web Worker.
5. **Decode.** `output0` is `[1, 4 + classes + 32, N]`: box, class scores,
   mask coefficients. Score = highest class score; keep scores above `conf`;
   per-class NMS at the config IoU. Masks: coefficients · prototypes
   (`output1`), upsampled bilinearly to the input size, thresholded at 0.5,
   then cropped to the box. That is the order Ultralytics 8.4.184 uses.
   Output names and shapes are read from the session, not assumed.
6. **Draw** masks and outlines back on the original photo.

Useful URL switches: `?threads=1`, `?worker=0`, `?model=model_fp32.onnx`,
`?match=0`, `?crop=0`.

## Parity check

```bash
python3 serve.py &                       # the app must be running
npm i playwright                         # once; or set PLAYWRIGHT_DIR
.venv/bin/python parity_check.py --pt models/runB_best.pt
```

It runs the bundled samples through Ultralytics (`model.predict(imgsz,
conf, iou)`), through the ONNX file in Python, and through the real app in
headless Chrome, and compares detections (same class and mask IoU ≥ 0.5, the
rule `eval_test.py` uses). It also compares against the predictions the
Workflow Studio saved for the same images. Results: `parity/parity_results.json`.

## 30-second demo recording

Before recording: open the app once so everything is cached and the pill says
"ready"; turn on Do Not Disturb; on the laptop open `http://127.0.0.1:8000`,
tap a diseased test leaf and untick **Masks**, so a plain leaf fills the screen.

| Time | Tap | Say |
| --- | --- | --- |
| 0–6 s | Nothing: show the top of the app (model size, "Runs on") | "This is the exact model our workflow trained: under 6 megabytes, running in the phone's browser. No server." |
| 6–12 s | **Healthy** test leaf | "A healthy leaf: no spots found." |
| 12–20 s | The apple scab thumbnail under the result | "A diseased leaf: every spot outlined, in a fraction of a second, on the phone." |
| 20–30 s | **Take a photo**, shoot the leaf on the laptop screen, **Use Photo** | "And a live photo. Trained in the Workflow Studio on a GPU cluster; it runs in the farmer's pocket." |

## Limits

- The model was trained and tested on lab-style close-ups of single leaves.
  Phone photos in the field are a different domain: treat results as a demo,
  not a diagnosis.
- First load downloads the WebAssembly runtime (about 11 MB uncompressed, from
  the jsDelivr CDN) and the model; both are cached by the browser afterwards.
