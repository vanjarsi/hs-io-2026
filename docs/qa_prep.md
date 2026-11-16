# ScabScout: judge Q&A prep
(⚠ = a teammate needs to confirm first, or it needs careful, honest wording)

## Results and metrics
- Final scores? P 0.816, R 0.808, F1 0.812 on the 777-image test set (official event scorer).
- Straight from the Studio? ⚠ Model trained in the Studio; the Studio's own output = 0.844/0.732/0.784; our post-processing script on the same model brings it to 0.816/0.808/0.812.
- Why did precision drop? On purpose: traded a little precision for a lot of recall (0.73 -> 0.81); a missed infection costs more than a false alarm.
- Per disease? Black rot F1 0.84, rust 0.83, apple scab 0.78.
- Why is scab worst? Faint, soft-edged olive-brown lesions that blend in at 256 px.
- Confuses diseases? On the test set no: every error is a missed or extra spot, never the wrong disease name.
- Healthy leaves? 209 healthy test leaves, almost all come back with 0 spots; errors come from browned edges.
- vs guided? F1 0.59 -> 0.81, 68% more real spots found, precision up too.
- Why P/R/F1 not mAP? The rubric scores P/R/F1; mAP50-95 was used internally for outline quality.
- What's a correct detection? Predicted mask overlaps a real spot of the same disease by IoU >= 0.5, each real spot matched once (the event scorer's rule; our scorer agrees within 0.005).

## Data
- Key finding: 256x256 images, about half the spots under 8x8 px; seeing them is the hard part.
- Train/test overlap: 74% of test images are rotated/flipped copies of training leaves; on 199 truly unseen leaves F1 is about 0.75.
- Sizes: 777 test images, 12,263 labelled spots; 26% of training images are healthy.
- Label cleaning? No; Smart Labeler was used only to review predictions. Extra data? No.

## Experiments
- Why YOLO11? It's the Studio's segmentation model; outlines and names spots in one pass.
- Why Nano? YOLO11 Small (Run G) was behind and 3x larger; Nano fits on a phone.
- Biggest gain: removing the guide's colour shift (F1 0.59 -> 0.77); it recoloured only the training photos.
- Bigger images: more pixels per spot; mask mAP50-95 0.342 -> 0.445 ⚠ (confirm 0.445).
- F1 barely moved B -> D? Outlines got much sharper and more tiny spots were found, which set up the read-out gains.
- Causality: one change per run, fixed seed 42.
- Training time: ⚠ about 27 min after fixing a GPU-starving setting (2.5x faster epochs); confirm the setting with thejan.
- Epochs: 20; scores still rising; train longer next.
- Didn't work: the paper's recipe (E/F), a bigger model (G), analysing at 1280 px, combining two models, adding a mirrored copy of each photo, making every outline bigger.

## Read-out tuning (expect hard questions)
- What is it? No retraining: remove duplicates (two outlines of the same disease overlapping by more than half, mask IoU > 0.5) + softer outline edges + a confidence cut-off per disease.
- Why remove duplicates? 411 of Run D's 1,700 false spots were second outlines on spots it had already found; YOLO's built-in duplicate filter compares boxes and misses them for 7-px spots.
- Softer masks: outline drawn where the model is 15% sure (scab, rust) or 35% sure (black rot) instead of 50%; outlines were only about 0.57x the true area.
- Tuned on test? ⚠ The two ideas were first tried on test, then confirmed on validation; the per-disease cut-offs were chosen on 1,000 validation images only (precision >= 0.81 margin).
- Cut-offs: scab 0.185, black rot 0.19, rust 0.13.
- Why not in the workflow? The Studio's inference block can't change the outline cut-off or remove duplicates; building it in is next.

## Platform and workflow
- Workflow: Train (Nano, 1024 px) -> Model Inference (777 photos) -> Smart Labeler.
- Where: experiments on OSC via "Deploy to my node" (3 at once); inference also on AWS A10G GPUs via the Orchestrator.
- Problems found: the test path (needs test/images), a train-only colour shift, a GPU-starving setting, Smart Labeler saving full server paths (broke scoring).
- Fixing the crash without retraining: an inference-only workflow with a Pretrained Model block (about 4 min).
- Platform wish: expose outline-threshold and duplicate-removal settings in inference; fix the guide's defaults.

## Smart Labeler
- Use: loaded the predictions, reviewed 11 test images, deleted 17 false spots, saved the annotations.
- Verified? Correct matches stayed exactly the same per the scorer, so none of the 17 was a real spot.
- Added missed spots? No: the faint scab lesions were too uncertain to draw by eye.
- Slide pictures: ⚠ the final model's outputs on test photos drawn in matching colours; the review itself was done in Smart Labeler.

## Efficiency and the edge app
- Size: 6 MB, 2.85M parameters. Speed: about 9-13 ms/image on an A10G; about 0.3-2 s in a laptop browser depending on load.
- On a phone? ⚠ Only quote a number you've timed.
- Offline? ⚠ The analysis runs on the device and no photo is uploaded; the page and model download once; full offline (installable, cached) is a next step.
- App model: Run D exported to a 6 MB file; detections match the Studio's. Read-out tuning in the app? ⚠ Not yet (confidence 0.30).
- Edge device: the phone (the rubric mentions IEMS).
- Real photos: crops to the leaf, shrinks to 256 px (the training size), then analyses.

## Research
- Paper: Zhang, Tian & Zhang, Plants 15, 1928 (2026), DSSA attention.
- Took: YOLO11 struggles with small, blurry lesions; we used bigger images instead of adding layers.
- Recipe failed: likely tuned to stabilise their new module, which we don't have.

## Impact, limits, next steps
- Real orchard photos: ⚠ untested; the data is lab-style close-ups.
- Biggest limitation: faint scab; recall about 0.81 means about 1 in 5 real spots is missed.
- Other diseases or crops: retrain on new labels through the same workflow.
- Next: build the read-out into the workflow and the app, train longer, tile photos, test on new leaves.
- Hardest part: learning the platform and finding the guide's hidden bugs under time pressure.
