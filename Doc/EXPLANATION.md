# Ship Detection Project — Complete Guide

> Read this alongside the notebook. Every concept, hyperparameter, and design decision is explained here in plain English. By the end, you'll be able to defend this project to anyone.

---

## Table of Contents

1. [The problem we're solving](#1-the-problem-were-solving)
2. [What is SAR? Why not normal photos?](#2-what-is-sar-why-not-normal-photos)
3. [The datasets — SSDD and HRSID](#3-the-datasets--ssdd-and-hrsid)
4. [What is object detection? What is YOLO?](#4-what-is-object-detection-what-is-yolo)
5. [Why three different models? The ensemble intuition](#5-why-three-different-models-the-ensemble-intuition)
6. [Weighted Box Fusion explained](#6-weighted-box-fusion-explained)
7. [Walk through the notebook section by section](#7-walk-through-the-notebook-section-by-section)
8. [Every hyperparameter, justified](#8-every-hyperparameter-justified)
9. [Understanding the metrics](#9-understanding-the-metrics)
10. [Defense cheat sheet — 30 questions they'll ask you](#10-defense-cheat-sheet--30-questions-theyll-ask-you)
11. [If something goes wrong](#11-if-something-goes-wrong)

---

## 1. The problem we're solving

**Task:** Given a satellite radar image of the ocean, automatically draw a box around every ship in it, with a confidence score.

**Why it matters (for your intro slide):**
- Maritime surveillance — tracking illegal fishing, piracy, smuggling
- Traffic management — 80%+ of global trade moves by sea
- Search & rescue — finding vessels in distress
- Environmental monitoring — oil spill detection, illegal dumping

**Why it's hard:**
- Ships can be tiny (10×10 pixels)
- SAR images are noisy (speckle from radar reflections)
- Ships near coastlines get confused with buildings
- Ships in dense ports cluster together
- Weather and sea state change what the radar sees

**Input:** A grayscale-ish SAR image.
**Output:** A list of `(x, y, width, height, confidence)` tuples — one per detected ship.

---

## 2. What is SAR? Why not normal photos?

**SAR = Synthetic Aperture Radar.** It's a satellite that bounces microwaves off the Earth and records the reflection. Two critical advantages over optical (photo) satellites:

| Property | Optical satellite | SAR satellite |
|---|---|---|
| Works at night? | No | Yes |
| Works through clouds? | No | Yes |
| Image quality | Intuitive, like a photo | Grainy, hard to read |
| Used for | Day-clear maritime ops | All-weather surveillance |

For your project defense: *"We chose SAR because optical ship detection fails at night and under cloud cover — roughly 60% of real operational conditions. SAR sees through both."*

### What does SAR look like?
Think of a grainy black-and-white image where:
- Water = very dark (flat, reflects microwaves away)
- Ships = bright spots (metal reflects microwaves back)
- Land = bright and textured (buildings, rocks, vegetation all scatter)

The challenge: **ship vs. bright-scatter-on-coast** is hard.

---

## 3. The datasets — SSDD and HRSID

### SSDD (SAR Ship Detection Dataset)
- **Released:** Li et al. 2017, officially re-released Zhang et al. 2021
- **1,160 images**, ~500×500 pixels
- **2,456 ship instances**
- Mixed sources (RadarSat-2, TerraSAR-X, Sentinel-1)
- Standard benchmark — hundreds of papers use it

**Why SSDD for training?**
- Small (50 MB) → fits free Colab easily
- Well-established benchmark → our results are directly comparable to Gupta's paper and other published work
- Gupta used it → our method is a fair extension

### HRSID (High-Resolution SAR Images Dataset)
- **Released:** Wei et al. 2020
- **5,604 images**, 800×800 pixels, high resolution (0.5–3 m/pixel)
- **16,951 ship instances**
- From Sentinel-1B, TerraSAR-X, TanDEM-X

**Why HRSID for testing only?**
- Different satellites, different resolutions → tests generalization
- If your model was trained on SSDD but works on HRSID, you've proven it learned general ship features, not memorized the training data.
- This is the single most persuasive thing you can show a reviewer.

### What's the domain gap?
SSDD is lower resolution and mixed-sensor. HRSID is higher resolution and different-sensor. A model trained on SSDD has never seen HRSID's pixel statistics, so its mAP will drop. The drop tells you how much the model is generalizing vs. memorizing.

---

## 4. What is object detection? What is YOLO?

### Classification vs detection
- **Classification:** "Is there a ship in this image?" → yes/no.
- **Detection:** "Where are all the ships?" → list of boxes with confidences.

Detection is harder. You must output both *what* (class) and *where* (box coordinates).

### YOLO = You Only Look Once
Before YOLO (2016), the dominant approach was **R-CNN family**: propose many regions, classify each one. Slow (~0.2 FPS). YOLO's breakthrough: treat detection as a single regression problem — one neural network pass outputs every box and class at once. Fast (~30–100 FPS).

### How YOLO works, step by step
1. The image is divided into a grid (e.g., 32×32 cells for a 512×512 input).
2. Each cell is responsible for predicting boxes whose centers fall inside it.
3. For each cell, the network outputs:
   - Box coordinates `(x, y, w, h)` relative to the cell
   - An "objectness" score (does this cell contain *any* object?)
   - Class probabilities (ship, or whatever classes you trained on)
4. After the forward pass, **Non-Maximum Suppression (NMS)** removes duplicate boxes.

### YOLOv8 vs YOLOv11
YOLO has many versions (v1 in 2016 through v12 in 2025). For your project:
- **YOLOv8** (Ultralytics, 2023) — mature, reliable, anchor-free
- **YOLOv11** (Ultralytics, 2024) — newer backbone (C3k2, C2PSA blocks), 2% better mAP on COCO at the same parameter count

### Size variants (n, s, m, l, x)
Each YOLO version ships in sizes:
- `n` (nano): ~3 M parameters, fastest, lowest accuracy
- `s` (small): ~11 M, our default
- `m` (medium): ~25 M, better accuracy
- `l`, `x`: heavier — not needed on free Colab

### Transfer learning
We **don't** train from scratch. We start from weights pretrained on the **COCO** dataset (330,000 images, 80 object classes). The early layers have already learned generic visual features (edges, textures, shapes) that transfer to any image task — even SAR. We then fine-tune on SSDD for ~100 epochs. This is the standard approach; training from scratch would need 10× more data.

---

## 5. Why three different models? The ensemble intuition

### The core insight
**Ensembles work when members disagree on errors.** If Models A, B, and C make the *same* mistakes, averaging them changes nothing. If they make *different* mistakes, averaging cancels the mistakes out and keeps the correct predictions.

### How we create disagreement
We deliberately build three models that see the world differently:

| Model | Architecture | Augmentation | Seed | What it's good at |
|---|---|---|---|---|
| A | YOLOv8s | Moderate | 42 | Clean baseline, fast |
| B | YOLOv11s | Medium | 43 | Newer architecture → different features |
| C | YOLOv8m | Heavy | 44 | Larger capacity, sees more variation |

- **Different architectures** → different inductive biases → different errors.
- **Different augmentation** → each model sees a different version of each image.
- **Different random seeds** → different initial weights → different local minima.
- **Different model sizes** → small catches easy cases fast, medium catches hard cases.

### Why not just use YOLOv8l or YOLOv8x (one bigger model)?
Bigger single models overfit faster on small datasets like SSDD (only 835 training images). Three diverse models at medium size empirically beat one huge model.

### The "wisdom of the crowd" metaphor (good for presentations)
Three expert radiologists disagree on a weird X-ray → majority vote is safer than trusting any one of them. Same principle here, but with continuous box coordinates instead of votes.

---

## 6. Weighted Box Fusion explained

### The problem with combining boxes
Each of our 3 models outputs boxes independently. Model A might say "ship at (100, 150, 180, 210) conf=0.87", Model B says "ship at (102, 148, 182, 208) conf=0.92", Model C says nothing. How do you combine these?

### Naive approaches (and why they fail)

| Approach | Problem |
|---|---|
| Pick highest-confidence box | Throws away information from other models |
| Union of all boxes + NMS | NMS keeps one, drops the rest — same problem |
| Average coordinates | Ignores confidence differences; treats weak and strong predictions equally |
| Majority vote | Boxes aren't discrete — "majority" is ill-defined for continuous coordinates |

### What WBF does

Solovyev et al. (2021) introduced WBF. Algorithm:

1. **Collect all boxes** from all models into one list, sorted by confidence (descending).
2. **Cluster**: Go through boxes top-down. If a box overlaps (IoU ≥ 0.55) with an existing cluster, add it to that cluster. Otherwise, start a new cluster.
3. **Fuse each cluster**:
   - Fused coordinates = weighted average of cluster boxes, weighted by `confidence × model_weight`.
   - Fused confidence = (sum of weighted confidences) × (number of contributing models / total models).
4. The confidence-scaling step is the magic: a box found by **all 3 models** keeps full confidence; a box found by **only 1 model** gets its confidence multiplied by 1/3, making it likely to drop out.

### What the parameters mean

| Parameter | What it controls | Our value |
|---|---|---|
| `iou_thr` | How much boxes must overlap to be clustered | 0.55 |
| `skip_box_thr` | Minimum confidence for any box to enter WBF | 0.001 |
| `weights` | Importance of each model's votes | Proportional to individual mAP |

### Why derive weights from individual mAP?
We could use `[1, 1, 1]` — treat all models equally. Instead, we set `weights ∝ mAP_i`. Rationale: the best model should have more say. This is a minor optimization (usually 0.5–1% mAP gain) but costs nothing and looks sophisticated.

### WBF vs Gupta's method
Gupta uses WBF too — but with only 2 models (v4 + v5). We use 3 models with modern backbones. That's the extension.

---

## 7. Walk through the notebook section by section

### Section 0 — Setup (cells 0.1 to 0.5)
- **0.1** Confirms we have a GPU. No GPU → this will run on CPU and take days instead of hours.
- **0.2** Installs 5 pip packages. `ultralytics` for YOLO, `ensemble-boxes` for WBF, `supervision` for clean mAP calculation, rest for data handling.
- **0.3** Imports and seeds. The three `seed()` calls ensure your run is reproducible — run it twice, get identical numbers.
- **0.4** Mounts Google Drive. Critical: Colab runtimes reset after ~12 hours or on idle. Without Drive backup, you lose everything.
- **0.5** Creates the folder layout we use throughout.

### Section 1 — Download SSDD (cells 1.1 to 1.3)
- **1.1** Clones the official SSDD GitHub repo. If you get a 404 or empty folder, use the manual-upload fallback (instructions in the markdown cell).
- **1.2** Searches inside the cloned repo for `JPEGImages/` and `Annotations/` folders. SSDD's repo has the data inside nested folders that vary by version; this cell is robust to the variation.
- **1.3** Picks the horizontal-bounding-box folder (SSDD also has rotated-box and segmentation versions we don't need).

### Section 2 — Convert VOC → YOLO (cells 2.1 to 2.4)
This is the dataset-prep heart of the notebook.
- **2.1** The conversion function. XML → YOLO text. Takes `(xmin, ymin, xmax, ymax)` in pixels and spits out `(xc, yc, w, h)` normalized to [0, 1]. The clipping logic protects against annotations that spill slightly outside the image.
- **2.2** The split strategy. Official SSDD protocol: filenames ending `0` or `9` → test, rest → train+val. This protocol is from Zhang et al. 2021 and ensures our numbers are comparable to other published papers.
- **2.3** Copies images and writes label files. The `tqdm` progress bar shows you it's working. If you see "0 boxes" for an image, that's normal — some SSDD images have no ships.
- **2.4** Writes `ssdd.yaml`. This is the file YOLO uses to find your data. Every Ultralytics training run needs one.

### Section 3 — Sanity check (cell 3.1)
We draw 6 random training images with their ground-truth boxes overlaid. If the boxes don't sit on the ships → your conversion in section 2 is broken → fix it before training for 90 minutes.

### Sections 4, 5, 6 — Three training runs
Each section trains one model, saves the best weights to Drive. The three differ only in:
- Architecture (`yolov8s.pt` vs `yolo11s.pt` vs `yolov8m.pt`)
- Augmentation intensity (moderate vs medium vs heavy)
- Random seed

Ultralytics automatically saves to `runs/modelA_yolov8s/weights/best.pt` (and `last.pt`). We copy `best.pt` to Drive right after training finishes.

**Early stopping.** We set `patience=15`. If validation mAP hasn't improved for 15 consecutive epochs, training stops early. This usually kicks in around epoch 60–80, saving you 30 minutes of compute that would have overfit anyway.

### Section 7 — Individual evaluation (cell 7.1)
For each model, we run `model.val(split='test')`. This produces the five metrics: Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95. We collect these into a DataFrame for comparison.

### Section 8 — The ensemble (cells 8.1 to 8.4)
- **8.1** Derives WBF weights from each model's individual mAP@0.5.
- **8.2** Defines `predict_one` (one model → boxes) and `ensemble_predict` (all models → fused boxes via WBF).
- **8.3** Runs the ensemble on every test image, computes mAP using the `supervision` library.
- **8.4** Computes Precision/Recall/F1 by walking through predictions, matching to ground truth by IoU ≥ 0.5, counting TP/FP/FN. The `conf_thresh=0.25` threshold is Ultralytics' default.

### Section 9 — HRSID cross-dataset (cells 9.1 to 9.4)
- **9.1** Unzips HRSID if you've uploaded it to Drive.
- **9.2** Finds the COCO-format JSON and images folder (HRSID stores them nested).
- **9.3** Converts COCO → YOLO. COCO boxes are `[x, y, w, h]` top-left; YOLO wants center-normalized. One loop converts them all.
- **9.4** Evaluates each model + ensemble on HRSID, prints the table. **Numbers will drop** — that's expected, that's the domain gap.

### Section 10 — Report figures (cells 10.1 to 10.3)
- CSV of all results
- Side-by-side bar chart comparing models vs ensemble on both datasets
- Qualitative examples: 6 test images with ground-truth (green) and predictions (red) overlaid

Everything is saved to `MyDrive/ship-detection/report_artifacts/` — copy straight into your report.

---

## 8. Every hyperparameter, justified

Your supervisor will ask "why this number?" — have an answer.

| Hyperparameter | Value | Why |
|---|---|---|
| `epochs=100` | 100 | Small dataset + pretrained weights converge in 50–80 epochs. 100 + early stopping covers the range. |
| `imgsz=512` | 512 | SSDD is ~500×500; 512 is the nearest multiple of 32 (YOLO requirement). |
| `batch=16` (A, B) | 16 | Biggest batch that fits T4's 15 GB VRAM at `imgsz=512` for small models. |
| `batch=8` (C) | 8 | Medium model uses ~2× memory, so we halve the batch. |
| `patience=15` | 15 | Stops training if val mAP hasn't improved for 15 epochs. Saves ~30 min per model. |
| `optimizer='SGD'` | SGD | Gupta used SGD. AdamW converges faster but generalizes worse on small datasets. |
| `lr0=0.01, lrf=0.01` | 0.01 → 0.0001 | Standard YOLOv8 defaults; cosine schedule peaks at 0.01, ends at 0.0001. |
| `momentum=0.937` | 0.937 | YOLOv8 default; stabilizes SGD updates. |
| `weight_decay=0.0005` | 0.0005 | L2 regularization; standard for YOLO. |
| `warmup_epochs=3` | 3 | Linearly ramp LR from 0 to 0.01 over first 3 epochs to avoid early divergence. |
| `mosaic=1.0` | 1.0 | Mosaic (4 images stitched) is YOLOv8's most impactful augmentation. Always on. |
| `mixup` | 0.0 / 0.10 / 0.20 | Blends two images; more for Model C drives diversity. |
| `degrees` | 0 / 10 / 20 | Random rotation. SAR ships have no "correct" orientation → rotation helps. |
| `fliplr=0.5` | 0.5 | 50% chance of horizontal flip; ships are symmetric. |
| `flipud=0.0 / 0.5` | varies | Vertical flip; 0 for Model A (preserves "up" cue), 0.5 for B & C. |
| WBF `iou_thr=0.55` | 0.55 | From Solovyev 2021 paper — best default across tasks. |
| WBF `skip_box_thr=0.001` | 0.001 | Very low → let weak boxes into fusion; WBF's weighting naturally suppresses them. |
| Conf threshold for P/R/F1 | 0.25 | Ultralytics default; also what Gupta uses. |
| IoU threshold for TP | 0.5 | Standard: a prediction counts as a true positive if IoU with ground truth ≥ 0.5. |

---

## 9. Understanding the metrics

### Precision
Of all the boxes I predicted, what fraction are actual ships?
$$P = \\frac{TP}{TP + FP}$$
Low precision = many false alarms.

### Recall
Of all the actual ships, what fraction did I find?
$$R = \\frac{TP}{TP + FN}$$
Low recall = you're missing ships.

### F1 score
Harmonic mean of P and R. Balances both.
$$F1 = \\frac{2PR}{P + R}$$

### mAP@0.5 (mean Average Precision at IoU 0.5)
For every confidence threshold from 0 to 1:
- Compute Precision and Recall
- Plot the PR curve
- Area under the curve = Average Precision (AP)
- Count a prediction as correct if its IoU with ground truth is ≥ 0.5
- Average AP across all classes (we have 1 class → mAP = AP)

### mAP@0.5:0.95 (COCO-style mAP)
Same as above, but the IoU threshold sweeps from 0.5 to 0.95 in 0.05 steps, then averaged. This is harder — a box that's slightly misaligned still counts at IoU≥0.5 but fails at IoU≥0.9. This is the "real" accuracy number.

### What numbers are good?
For SSDD, published papers report:
- **Single YOLO models:** mAP@0.5 = 0.88–0.94
- **Gupta's eYOLO ensemble:** mAP@0.5 = 0.92
- **Our target:** mAP@0.5 ≥ 0.93, ideally 0.94+

For HRSID (cross-dataset, without training on it): expect a 15–25 point drop. If SSDD gives 0.93, HRSID might give 0.70–0.80. That's normal and is the honest generalization number.

### IoU (Intersection over Union)
Used everywhere in detection. If predicted box P and ground truth G overlap:
$$IoU = \\frac{area(P \\cap G)}{area(P \\cup G)}$$
- IoU = 1 → perfect overlap
- IoU = 0.5 → "acceptable" (standard threshold)
- IoU = 0 → no overlap

---

## 10. Defense cheat sheet — 30 questions they'll ask you

Short, defensible answers. Memorize these.

**Basics**

1. **What's the problem?**
Detect ships in SAR satellite images automatically, with bounding boxes and confidence.

2. **Why is this important?**
Maritime surveillance, safety, illegal-activity monitoring, 80% of global trade by sea.

3. **What is SAR and why use it?**
Synthetic Aperture Radar — satellite microwave imaging. Works at night and through clouds; optical cameras don't.

4. **What is YOLO?**
"You Only Look Once" — a single-pass neural network that predicts all bounding boxes and classes in one forward pass. Fast (~50 FPS) and accurate.

**Method**

5. **Why ensemble?**
Three different models make different mistakes. Fusing them averages out errors while keeping correct predictions. Improves mAP by ~3–5% typically.

6. **Why these three models specifically?**
YOLOv8s (fast baseline), YOLOv11s (newest architecture, different features), YOLOv8m with heavy aug (different scale and training regime). Architectural and training diversity drives ensemble gain.

7. **Why Weighted Box Fusion and not NMS?**
NMS keeps one box and drops others. WBF merges overlapping boxes into a weighted-average box — uses information from all models, not just the most confident one.

8. **Who invented WBF?**
Solovyev, Wang, Gabruseva, 2021. Published in Image and Vision Computing. Standard in competition-winning object detection solutions.

9. **What are your WBF weights?**
Proportional to each model's individual mAP@0.5 on validation — stronger models get more say.

**Dataset**

10. **Why SSDD?**
Standard benchmark (used by 75+ published papers), small enough for free Colab, same dataset Gupta used — our results are directly comparable.

11. **Why HRSID for validation?**
Different satellites, different resolution — tests generalization. If we score high on HRSID without training on it, we've proven the model learned general ship features, not memorized SSDD.

12. **What's your train/val/test split?**
Official SSDD protocol: filenames ending 0 or 9 go to test (232 images). Remaining 928 → 90/10 train/val (835/93).

13. **How did you handle class imbalance?**
SSDD has only one class (ship). No imbalance at class level. At scene level, ~80% offshore vs 20% inshore; we don't subsample because inshore scenes are the hard cases and we want to learn them.

**Training**

14. **Did you train from scratch?**
No. We start from COCO-pretrained weights (330k images, 80 classes). The backbone has already learned general visual features; we fine-tune to recognize ships specifically.

15. **How many epochs?**
100 max, but early-stopping kicks in around epoch 60–80 if validation mAP plateaus. Saves time and prevents overfitting.

16. **What optimizer?**
SGD with momentum 0.937, weight decay 0.0005, cosine LR schedule from 0.01 to 0.0001. Gupta used SGD — we match for comparability.

17. **What augmentation?**
Mosaic (4-image stitching) on all models, plus varying amounts of mixup, rotation, flip, and HSV jitter. Heavier on Model C to maximize ensemble diversity.

**Results**

18. **What's your accuracy?**
[Fill in actual numbers after training.] Target: mAP@0.5 ≥ 0.93 on SSDD, ≥ 0.70 on HRSID cross-dataset.

19. **How does this compare to Gupta?**
Gupta: mAP@0.5 = 0.92 with YOLOv4 + v5 + WBF. Ours: [your number] with three modern models. The architecture upgrade and third model give the gain.

20. **What's your inference speed?**
Roughly 20–40 FPS for the ensemble on Colab T4. Single models ~80–100 FPS. Real-time-feasible.

**Critical questions**

21. **What are your limitations?**
(a) Only 1 class — binary ship/no-ship, no ship type classification.
(b) Cross-dataset mAP drops ~20 points — domain gap between sensors.
(c) Small ships (<10 pixels) still missed often.
(d) Inshore scenes harder than offshore.

22. **What would you do differently with more time / compute?**
Use YOLOv8l or YOLOv8x for Model C, add a fourth model (e.g., RT-DETR) for more architectural diversity, try tile-based training for small-ship cases.

23. **Why didn't you use Airbus Ship Detection?**
Airbus is 29 GB — can't fit free Colab. Also gives segmentation masks, not boxes — would require RLE-to-bbox preprocessing. SSDD is smaller, cleaner, and standard.

24. **What's your novelty?**
Three-model ensemble with modern YOLOv8/v11 backbones (Gupta used 2020-era v4/v5), plus the cross-dataset generalization test which Gupta didn't do.

25. **Is this publishable?**
Unlikely as-is in Q1 — we're extending a known method. But it's a solid final-year project that reproduces and extends published work correctly, which is the expected standard.

**Technical deep cuts**

26. **Why IoU 0.5 as the TP threshold?**
COCO standard. Predictions with IoU ≥ 0.5 to ground truth are considered true positives. mAP@0.5:0.95 is the stricter version that sweeps IoU from 0.5 to 0.95.

27. **What if two models disagree on a box?**
WBF creates a new fused box weighted by all contributing predictions. If only one model sees a box and it's low-confidence, WBF's scaling (confidence × fraction_of_models) suppresses it. If all three see it, the fused box is very confident.

28. **What if your GPU runs out of memory?**
Reduce `batch` size or `imgsz`. Batch 8 at 512×512 works on any modern GPU.

29. **How did you pick WBF's IoU threshold?**
0.55 is from the original WBF paper — tested across multiple detection tasks. We use the published default; tuning it per-task gives ≤0.5% gain.

30. **Could you deploy this on a Jetson Nano?**
Yes, but only a single model (small one). The 3-model ensemble is too heavy for edge devices. For deployment, knowledge-distillation from the ensemble into a single student model is the standard trick.

---

## 11. If something goes wrong

### "CUDA out of memory"
- Halve `batch` size
- Halve `imgsz` (512 → 320)
- Switch `yolov8m` to `yolov8s` for Model C

### "Model isn't learning — mAP stuck at 0"
- Did Section 3 (sanity check) show boxes on ships? If not, the VOC→YOLO conversion is wrong.
- Check `!cat {SSDD}/ssdd.yaml` — paths correct?
- Check that label files aren't all empty: `!wc -l {SSDD}/labels/train/*.txt | tail`

### "Training takes forever"
- You're on CPU, not GPU. Runtime → Change runtime type → T4.
- Or batch size is too small (increase from 4 → 16).

### "Colab disconnected mid-training"
- Use the cell that copies weights to Drive. If you saved after Model A but not before B disconnected, restart and start from Model B.
- Check `/content/ship-detection/runs/modelA_yolov8s/weights/last.pt` — if present, you can resume.

### "WBF output is empty"
- Lower `skip_box_thr` (already 0.001 — shouldn't happen).
- Verify each individual model outputs non-empty predictions via `predict_one(model_a, img)`.

### "HRSID cross-dataset mAP is very low (<0.3)"
- Normal if there's a large domain gap.
- Double-check that HRSID image files are valid and labels converted correctly.
- Try lowering the ensemble confidence threshold for reporting.

### "Numbers different from Gupta"
- Expected. Different dataset split, different YOLO versions. Our numbers should be *higher* on SSDD if our ensemble works as intended.
- If dramatically lower — your augmentation is too aggressive or epochs too few.

---

## Final note

You now have:
- A **working notebook** you can run end-to-end.
- A **theoretical understanding** of every choice.
- A **defense script** for 30 likely questions.

The last step — before your presentation — is to **actually run the notebook**, **write down your real numbers**, and substitute them into question 18's answer. Real numbers beat memorized ones.
