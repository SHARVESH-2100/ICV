# AI Product Authentication Using Logo & Packaging Recognition

**Detect counterfeit product packaging from a single photo.**

A **team mini-project** for the **Introduction to Computer Vision (ICV)** course. The system compares
a photo of a product package against a library of authenticated reference images for that product,
then decides whether the package looks genuine or counterfeit. It does this by recognising the
product's **logo and packaging layout** and running eight independent checks — keypoint alignment,
layout matching, colour statistics, print-quality analysis, surface noise, an embedded security QR
code, and a learned Siamese CNN signature — fused into one 0–100 authenticity score with a
human-readable explanation.

The project ships **no dataset**. You create your own reference library from photos of real
products, and the code generates training and test data from it automatically. See
[Dataset — create your own](#dataset--create-your-own).

Two interfaces are included: a **Streamlit app** (`pro.py`) and a dependency-free **single-page web
app** (`web/server.py` + `web/index.html`).

> 🎓 Developed as a group mini-project for the **Introduction to Computer Vision (ICV)** subject,
> applying the techniques covered in the course — keypoint detection and matching, homography
> alignment, colour and edge analysis, image-quality forensics, QR decoding, and deep metric learning
> — to a concrete, working application.

---

## Academic context

This project was built for coursework, not for commercial deployment. It is deliberately scoped so
that every stage is inspectable and explainable: each signal is a separate, tunable function with a
visible score, which makes it easy to demonstrate *why* a package is classified as genuine or
counterfeit.

|                     |                                                                 |
| ------------------- | --------------------------------------------------------------- |
| **Project title**   | AI Product Authentication Using Logo & Packaging Recognition    |
| **Project type**    | College mini-project — team/group work                          |
| **Subject**         | Introduction to Computer Vision (ICV)                           |
| **Domain**          | Product authentication / counterfeit detection                  |
| **Core techniques** | ORB + homography alignment, histogram & edge correlation, Laplacian/noise print forensics, QR verification, Siamese networks with contrastive loss, Grad-CAM explainability |
| **Deliverables**    | Streamlit app, standalone web app, training pipeline, evaluation script |

**Learning outcomes demonstrated**

- Feature-based image registration and why geometric alignment must precede image comparison.
- Choosing and normalising complementary similarity metrics, then fusing them with weights.
- Building a labelled dataset when none exists, via synthetic counterfeit generation.
- Contrastive (Siamese) representation learning on a small, self-supervised dataset.
- Producing explainable outputs — per-signal scores, reasons and heatmaps — instead of a bare label.

**Team work:** This project was completed as team work for the Introduction to Computer Vision
course.

> Note: the repository is named **ICV** after the **subject code** (Introduction to Computer Vision),
> not after the project — the project name is *AI Product Authentication Using Logo & Packaging
> Recognition*.

---

## The problem

Counterfeiting is cheap: scan a label, reprint it at lower resolution on cheaper stock, and the
package looks *almost* right. Human inspection catches the obvious fakes but struggles once the
print quality is merely mediocre. This project targets exactly that gap — the differences that
survive a casual glance:

- the logo doesn't line up with the genuine print geometry,
- colours drift because the reprint used different inks,
- edges are softer and the surface noise profile changes,
- the security QR code is missing, unreadable, or carries a different payload,
- the overall "visual signature" of the packaging is simply not the same object.

## Highlights

- **Logo & packaging recognition** — the logo/print features and overall packaging layout are the
  two most heavily weighted signals.
- **Multi-signal scoring** — eight independent checks fused with configurable weights.
- **Explainable verdicts** — per-signal scores, plain-English reasons, and a suspicious-region
  heatmap (Grad-CAM when a model is loaded, grid-deviation map otherwise).
- **Learned + classical** — runs out of the box with no trained checkpoint; improves once a Siamese
  encoder is trained on your products.
- **Self-supervised training data** — counterfeit examples are synthesised on the fly from genuine
  photos, so no labelled fake dataset is required.
- **Live reference library** — add products and reference photos from the UI; nothing to hard-code.
- **Local and private** — everything runs on your machine; images never leave it.

---

## How it works

```
                ┌────────────────────────┐
  reference ───►│  ORB keypoint aligner  │──┐
  image         └────────────────────────┘  │
                                            ▼
  query photo ──────────────► aligned query image
                                            │
        ┌───────────────────────────────────┼───────────────────────────────────┐
        ▼               ▼          ▼        ▼          ▼           ▼             ▼
   Packing layout   Logo/print   Colour   Edge      Sharpness   Surface     Security
   correlation      features     hist.    layout    (Laplacian) noise       QR code
        │               │          │        │          │           │             │
        └───────────────┴──────────┴────────┼──────────┴───────────┴─────────────┘
                                            ▼
                                 Siamese CNN embedding
                                  (learned signature)
                                            │
                                            ▼
                          weighted fusion → 0–100 score
                                            │
                          verdict + reasons + heatmap overlay
```

### Step by step

1. **Align.** ORB keypoints are matched between the query photo and each stored reference, and a
   homography warps the query onto the reference's geometry. This removes rotation, scale and
   viewpoint differences so the remaining checks compare like with like. If too few keypoints match,
   the code falls back to a plain resize instead of failing.
2. **Measure.** The aligned pair is scored on layout correlation, logo/print feature matches, HSV
   colour histograms, Canny edge structure, Laplacian sharpness ratio, and surface-noise ratio.
3. **Check the security code.** Any QR code on the reference and on the query is decoded and
   compared. Missing, unexpected, or different payloads are penalised.
4. **Embed (optional).** A Siamese encoder maps both images to a 128-D embedding; the Euclidean
   distance becomes the "learned visual signature" score. Skipped cleanly if no checkpoint exists.
5. **Fuse.** Each available signal is combined using the weights below, re-normalised over whatever
   is present, producing one score from 0 to 100.
6. **Explain.** The score maps to a verdict, weak signals are turned into readable reasons, and a
   heatmap highlights where the query deviates from the reference.

### Signals and weights

| Key            | Signal                      | Weight | What it measures                                                     |
| -------------- | --------------------------- | ------ | -------------------------------------------------------------------- |
| `template`     | Packaging layout match      | 0.20   | Normalised correlation of the aligned query against the reference     |
| `orb`          | Logo / print feature match  | 0.20   | Ratio of good ORB descriptor matches                                 |
| `color`        | Colour distribution         | 0.12   | HSV histogram correlation                                            |
| `edge_layout`  | Print edge structure        | 0.12   | Canny-edge correlation                                               |
| `sharpness`    | Print sharpness             | 0.10   | Laplacian-variance ratio vs reference (catches low-quality reprints) |
| `noise`        | Print surface noise         | 0.06   | Noise-floor ratio vs reference                                       |
| `qr`           | Security QR code            | 0.20   | Presence and payload equality of the embedded QR code                |
| `siamese`      | Learned visual signature    | 0.25   | Embedding distance from a contrastively trained encoder              |

### Verdicts

| Score      | Verdict          | Binary (`BINARY_THRESHOLD = 70`) |
| ---------- | ---------------- | -------------------------------- |
| ≥ 75       | Authentic        | Original                         |
| 55 – 74.9  | Likely Authentic | Original                         |
| 35 – 54.9  | Suspicious       | Fake                             |
| < 35       | Counterfeit      | Fake                             |

---

## Dataset — create your own

No images are committed to this repository. The system is **reference-driven**: it can only verify
products you have introduced to it. Building that library is the first thing to do after cloning.

### 1. Add genuine reference photos

Create one folder per product under `data/genuine/`, and drop in photos of a package you know is
genuine:

```
data/
└── genuine/
    ├── Acme Cola 330ml/
    │   ├── front.png
    │   ├── back.png
    │   └── side.png
    └── Acme Shampoo 500ml/
        ├── 01.jpg
        └── 02.jpg
```

The folder name **is** the product name shown in the UI. You can also do this from inside the app
(**Product Library → Add reference photos**), or with your phone's camera capture in the same tab.

### 2. Capture good references

Reference quality caps the accuracy of every signal, so a few minutes here pays off:

- **3–6 photos per product, minimum.** More angles beat more megapixels.
- **Show every face** — front, back, sides — plus one slightly angled shot of the logo area.
- **Fill the frame** with the package; avoid large empty backgrounds.
- **Even, diffuse light.** Kill glare: specular highlights break ORB alignment and skew colour.
- **Keep it sharp** and reasonably high-resolution (≈1000 px on the long side is plenty).
- **Mix in real-world variation** — different lighting and slightly different angles — so the
  library covers how the product actually appears in the wild.

### 3. Generate a test set

Once references exist, the repo can synthesise a labelled evaluation set for you:

```bash
python make_test_images.py
```

This reads every reference image and writes:

```
data/test/genuine/       # light augmentation of each reference  (expected: ORIGINAL)
data/test/counterfeit/   # simulated reprint of each reference   (expected: FAKE)
```

The counterfeit generator (`dataset.counterfeit`) reproduces the tell-tale signs of a fake:
downscale-then-upscale reprinting, colour/ink drift, saturation loss, JPEG artefacts, and optional
perspective warp. The genuine side applies only mild brightness/blur changes.

### 4. Evaluate

```bash
python run_test.py
```

Prints a per-image table of score, expected label, actual label and the reasons behind each verdict,
followed by overall accuracy at `BINARY_THRESHOLD = 70`.

### Building training data

Training needs no extra files. Every epoch samples your references and pairs each one with either a
light augmentation (positive) or a synthetic counterfeit (negative) — a self-supervised setup that
only requires genuine photos. Train on **your own** products before relying on the scores, since the
encoder learns the print characteristics of whatever it was shown.

> **Note on the included checkpoint.** `weights/siamese.pt` is a small checkpoint trained on the
> team's own reference set. It loads automatically if present, but delete it (or retrain) when
> working with different products — otherwise the learned signature is comparing against the wrong
> print characteristics.

---

## Project structure

```
ICV/                        # repository name = subject code
├── pro.py                 # Streamlit application (verify / library / train)
├── web/
│   ├── server.py          # stdlib HTTP server + JSON API
│   ├── index.html         # single-page front end
│   ├── css/style.css
│   └── js/app.js
├── config.py              # paths, weights, thresholds, model hyper-parameters
├── preprocess.py          # image decoding, letterbox, ORB-based Aligner
├── features.py            # template, ORB, colour, edge, sharpness, noise metrics
├── security.py            # QR detection and comparison
├── scoring.py             # Assessor: per-reference scoring, fusion, verdicts, reasons
├── explain.py             # Grad-CAM and heatmap overlay
├── dataset.py             # reference library I/O + synthetic counterfeit generator
├── models/
│   └── siamese.py         # Siamese encoder, contrastive loss, save/load helpers
├── train_script.py        # training loop (CLI + importable train_model)
├── make_test_images.py    # build a synthetic genuine/counterfeit test set
├── run_test.py            # batch evaluation with per-image output
├── data/
│   └── genuine/<product>/ # YOUR reference images (not committed)
├── weights/siamese.pt     # trained Siamese checkpoint
└── requirements.txt
```

---

## Quick start

```bash
# 1. install
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

# 2. add at least one product under data/genuine/<product>/  (see Dataset above)

# 3. run the Streamlit app
streamlit run pro.py

# ...or the standalone web UI
python web/server.py --port 8000     # → http://localhost:8000
```

### Verify a product

1. Pick a product from the library.
2. Upload a photo (or capture one with your webcam).
3. Press **Run verification** to get the verdict, per-signal breakdown, reasons and heatmap.

Until you add a product, the Verify tab shows an empty-state prompt pointing you to the library.

---

## Training the Siamese encoder

```bash
python train_script.py --epochs 25 --batch 32 --lr 5e-4
```

The same loop is available in the UI under **Train Siamese**, with a live loss chart. Checkpoints are
written to `weights/siamese.pt` and picked up automatically on the next run.

---

## Evaluation

```bash
python make_test_images.py    # synthesise data/test/{genuine,counterfeit}
python run_test.py            # score every test image
```

`run_test.py` reports per-image results and overall accuracy. The synthetic test set is a sanity
check, not a benchmark — validate against real captured counterfeits before trusting the numbers in
production.

---

## Configuration

All tunables live in `config.py`:

| Setting                                 | Purpose                                              |
| --------------------------------------- | ---------------------------------------------------- |
| `IMAGE_SIZE`, `SIAMESE_SIZE`            | Working / encoder input resolution                   |
| `WEIGHTS`                               | Contribution of each signal to the fused score       |
| `VERDICTS`, `BINARY_THRESHOLD`          | Score → verdict and score → Original/Fake boundaries |
| `ORB_FEATURES`, `ORB_RATIO_THRESHOLD`   | Alignment sensitivity                                |
| `SIAMESE_MARGIN`, `EMBED_DIM`           | Contrastive loss margin and embedding size           |
| `MODEL_PATH`, `GENUINE_DIR`, `DATA_DIR` | Filesystem locations                                 |

---

## Tech stack

`Python` · `OpenCV` · `PyTorch` · `Streamlit` · `NumPy` · vanilla HTML/CSS/JS for the web UI

---

## Limitations

- Reference quality drives everything — blurry or inconsistent references degrade every signal.
- Severe viewpoint change or glare can break ORB alignment; the fallback is a plain resize.
- Synthetic fakes are a proxy for real counterfeits; real-world validation and a larger reference
  library are recommended for production use.
- A model trained on one product family will not transfer well to another — retrain per catalogue.
- The web API has no authentication and is intended for local/trusted networks only.
- Built as an academic mini-project: it is a demonstration of the pipeline, not a deployable
  anti-counterfeiting product.

---

## License

MIT — see [LICENSE](LICENSE).
