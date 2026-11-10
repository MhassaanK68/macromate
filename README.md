# Macromate

**Recognition of eight Pakistani dish categories from photos using a partially fine-tuned CLIP model.**

Macromate is an academic prototype for a meal-logging application. This repository prepares the image dataset, trains and tests the recognizer, saves its weights, and runs predictions on new photos. The intended application lets users confirm a suggested dish and enter serving grams before calculating nutrition from a separate per-100-g reference.

**Implemented:** data preparation, training, checkpoint verification, held-out evaluation, reports, and CLI/Python inference. The graphical interface, serving-size workflow, nutrition integration, and daily log remain future work. There is no web server to launch.

## Deliverable 2 submission assets

- [Phase 3-4 report (PDF)](output/pdf/Deliverable_2_Report.pdf) and [LaTeX source](docs/model_report.tex).
- [Submission checklist and remaining requirements](docs/submission_checklist.md).
- [Git contribution evidence](docs/contributions.json).
- [Download evaluated model weights (54 MB ZIP)](https://github.com/MhassaanK68/macromate/releases/download/deliverable-2-weights/macromate-deliverable2-weights.zip) and [SHA-256 checksums](https://github.com/MhassaanK68/macromate/releases/download/deliverable-2-weights/SHA256SUMS.txt). Both downloads were verified without authentication. Extract the ZIP at the repository root; download the separate pinned base-model cache using step 3 below.

The model implementation is in models/. Run commands using python -m models.<command>.

Software validation is complete. Full submission readiness still requires Mohid's own coding commit and clarification of the historical manual-upload indicator; see the submission checklist.

## Contents

- [The simple workflow](#the-simple-workflow)
- [Run on your laptop](#run-on-your-laptop)
- [How the model is built](#how-the-model-is-built)
- [Train, test, and report](#train-test-and-report)
- [Run the final model](#run-the-final-model)
- [Results](#results)
- [Files and code](#files-and-code)
- [Troubleshooting](#troubleshooting)
- [Data and limitations](#data-and-limitations)
- [Team and guides](#team-and-guides)

## The simple workflow

```text
Prepared photos -> train -> best.pt + training.json
                           |
                           v
                        evaluate -> predictions.csv + evaluation.json
                                         |
                                         v
                                       report

New meal photo + best.pt + pretrained CLIP cache -> eight ranked dishes
```

The active `evidence/` folder contains exactly three files:

- **`training.json`:** settings, labels, epoch history, selected epoch, validation predictions, checkpoint identity, and verification.
- **`predictions.csv`:** one row per test photo, with the actual/predicted dish, complete class scores and rankings, and source attribution. List-valued cells use JSON so scores retain full precision.
- **`evaluation.json`:** evaluation status and protocol, metrics, confusion matrix, subgroup results, error reviews, and final model details.

The trained weights are stored separately at **`runs/phase4/best.pt`**. That adapted-weight checkpoint also needs the pinned pretrained base model in **`runs/hub/`**.

The existing model and results were consolidated without retraining or repeating test inference. Historical pilot selection and the evaluation correction remain under provenance in the three evidence files. The earlier historical archive is absent from this checkout; active model commands do not require it.

## Run on your laptop

### 1. Clone and create an environment

Use **Python 3.12** and Git. Initial installation and base-model download require internet access. Run all commands from the repository root.

```sh
git clone https://github.com/MhassaanK68/macromate.git
cd macromate
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

For a CPU laptop on Windows/Linux:

```sh
python -m pip install --upgrade pip
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For macOS, install `torch==2.9.0` from the default package index instead:

```sh
python -m pip install torch==2.9.0
python -m pip install -r requirements.txt
```

For a supported NVIDIA GPU on Windows/Linux, use the tested CUDA wheel index instead of the CPU index:

```sh
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Check the environment:

```sh
python -c "import sys, torch; print(sys.version); print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -m pip check
```

The recorded experiment used Windows 11, Python 3.12.14, PyTorch 2.9.0+cu128, and an RTX 3050 Laptop GPU with 4 GB VRAM. CPU execution is supported; Linux/macOS were not used for the recorded run. The commands below use `--device cpu`; substitute `--device cuda` where available.

`requirements.txt` installs the complete model and data environment through `requirements-model.txt`. Pillow 13.0.0 reproduces the original prepared PNG bytes; the historical training run used Pillow 11.3.0. All 400 decoded RGB arrays were checked to be identical between those versions.

### 3. Download the pinned base model once

```sh
python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/clip-vit-base-patch32', revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268', cache_dir='runs/hub', allow_patterns=['*.json', '*.txt', 'pytorch_model.bin'])"
```

The recognizer loads only this local revision. Keep the complete `runs/hub/` cache; subsequent loading can work offline.

### 4. Choose inference or training

**Use the existing model:** download the evaluated weights linked under submission assets and extract the ZIP at the repository root, then follow [Run the final model](#run-the-final-model). Fine-tuned weights and the base cache are excluded from Git; cloning does not supply them. Keep the separate pinned base-model cache as well.

**Train your own model:** prepare the included data and follow the next section. Use a fresh run and evidence directory so the existing results remain available.

### 5. Prepare data for training

The repository includes 400 source images and their manifests. Generate the prepared images and split locally:

```sh
python "data scripts/audit_sources.py"
python "data scripts/clean_data.py"
python "data scripts/preprocess.py"
python "data scripts/split_data.py"
python "data scripts/eda.py"
```

The scripts check source hashes, clean records, flag duplicates/outliers, create RGB 224 x 224 PNGs, assign class IDs, create a seeded split, and generate EDA figures. Source audit summaries are generated under `data/processed/source_audits/`, separate from model evidence. Near-duplicates require review; outlier flags do not automatically remove images. Record approved removals in `data/review_decisions.csv` before regenerating data.

Inspect `data/processed/split_summary.json` for **280 training / 64 validation / 56 test** photos and zero cross-split leakage. Check the model's manifest contract:

```sh
python -c "from collections import Counter; from models.data import load_records; rows, labels = load_records(); print(Counter(r['split'] for r in rows)); print(labels)"
```

Inference on a new photo does not need the training images or evidence files.

## How the model is built

The recognizer adapts pretrained **OpenAI CLIP ViT-B/32**, pinned to revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.

1. **Preprocess the photo:** correct EXIF orientation, convert to RGB, resize the shorter side to 224 with bicubic interpolation, center-crop to 224 x 224, divide pixels by 255, and normalize with CLIP's channel means/stds. Prepared PNGs are normalized once without a second crop or resize.
2. **Represent the labels:** the frozen text encoder converts `a photo of Pakistani <dish>` into one fixed, normalized 512-dimensional vector per dish.
3. **Represent the photo:** the vision encoder projects it into the same 512-dimensional space.
4. **Compare:** scaled dot products produce eight logits; softmax gives dish-ranking scores. No new classification head is added.
5. **Fine-tune:** update vision blocks 10 and 11, final vision layer normalization, and visual projection only: **14,570,496 trainable parameters**. The other layers and text encoder remain frozen.

The class-ID order is: **chicken biryani, chicken pulao, chicken karahi, daal, nihari, haleem, samosa, chapati** (IDs 0 through 7).

Training uses categorical cross-entropy without class weights or label smoothing, AdamW, 10% warmup followed by cosine decay, and gradient clipping at 1.0. Bias and normalization parameters receive no weight decay.

Defaults in `models/config.py` are the previously selected pilot settings: block learning rate **2e-5**, norm/projection learning rate **5e-5**, batch size **8**, weight decay **0.01**, maximum **20 epochs**, patience **5**, seed **42**, and dropout **0**. The historical six-candidate pilot comparison is preserved inside `training.json` under `provenance`; new training no longer requires separate pilot files.

## Train, test, and report

### Train

Use fresh directories for a new experiment:

```sh
python -m models.train --output runs/reproduction --evidence evidence/reproduction --device cpu
```

Training saves `runs/reproduction/best.pt` and updates `evidence/reproduction/training.json`. It selects the highest validation top-1 count, then lower loss, keeping the earlier epoch on exact ties. Five epochs without improvement trigger early stopping. Test images never enter the training loaders.

Training automatically reconstructs the saved checkpoint, compares all validation logits, and checks original-photo inference before marking the run completed. For a separate process check, optionally run:

```sh
python -m models.train --verify-only --evidence evidence/reproduction --device cpu
```

The verifier requires logit agreement within `1e-6` on the same device or `1e-4` across devices. For the supplied GPU-trained checkpoint, use `--device cuda` for strict replay. The submission CPU check agreed on all 64 validation top-1 labels but exceeded the strict batched-logit threshold (maximum difference 0.324503); CPU inference works, and the verifier continues to reject that numerical replay. A new CPU-trained run can be verified on CPU. Use `--epochs 1` for a short execution check, or `--config path/to/settings.json` for explicit settings. Such changes define a new experiment rather than the recorded full run.

### Evaluate

```sh
python -m models.evaluate --evidence evidence/reproduction --device cpu
```

This reads the checkpoint location from `training.json`, checks training/verification/data agreement, and records the evaluation protocol before inference. It then writes `predictions.csv` and completes `evaluation.json` with accuracy, precision/recall/F1, ROC-AUC, confusion counts, source/size subgroups, and model details.

The protocol and attempt status live inside `evaluation.json`; separate prepare/freeze commands are no longer needed. An existing evaluation, including a failed attempt, blocks repeated inference in that directory. Use saved predictions for analysis. A justified bug-fix rerun must preserve and document the prior attempt; repeated use of the same test set does not create new unbiased data.

### Report

To regenerate the existing report:

```sh
python -m models.report
```

For a new experiment:

```sh
python -m models.report --evidence evidence/reproduction
```

The report checks prediction hashes, label/ranking consistency, and exact metric reproduction from the CSV. It generates `docs/results.md`, `docs/model_report.tex`, and figures under `figures/model/`, without training or test inference. It can read the three evidence files without the model weights or prepared dataset; the error montage needs the original error photos.

Review misclassified photos and add observations to `error_reviews` in `evaluation.json`. The report explicitly marks missing observations as pending. Keep source labels unchanged unless a separately documented data review justifies a new experiment.

Report and figure destinations can be changed with `--output` and `--figures`; defaults are shared and replace generated outputs. Optional PDF compilation requires a TeX distribution with `pdflatex` on `PATH`. Create `output/pdf/` and run twice from the repository root:

```sh
pdflatex -interaction=nonstopmode -halt-on-error -jobname=Deliverable_2_Report -output-directory output/pdf docs/model_report.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=Deliverable_2_Report -output-directory output/pdf docs/model_report.tex
```

### Software checks

```sh
python "data scripts/verify_reproducible.py"
python scripts/verify_dataset.py
```

The data reproducibility command reruns the five data steps twice and compares generated CSV, JSON, PNG, and source-audit hashes. It regenerates data outputs intentionally. The dataset verifier checks the regenerated manifests and all 400 prepared images against the recorded experiment.

[Recorded software validation](docs/validation_results.json) documents the 22 regression checks, GPU training/checkpoint verification, and CPU inference checks completed for this deliverable. The regression test files are retained in Git history; the current project layout does not include a tests folder.

## Run the final model

CLI:

```sh
python -m models.infer runs/phase4/best.pt "path/to/meal.jpg" --device cpu
```

Replace the photo path with a real image. For a new experiment, use `runs/reproduction/best.pt`. Add `--prepared` only for a Phase 2 RGB 224 x 224 PNG; ordinary photos use the default preprocessing.

Python API:

```python
from models.infer import Recognizer

recognizer = Recognizer("runs/phase4/best.pt", device="cpu")
result = recognizer.predict("path/to/meal.jpg")
for suggestion in result["ranked_dishes"][:3]:
    print(suggestion["dish"], suggestion["score"])
```

Reuse the recognizer for multiple photos. It returns eight ranked entries with `label_id`, `dish`, raw `logit`, and softmax `score`, plus `scores_are_calibrated: false`. These are rankings among eight labels, not reliable confidence estimates or unfamiliar-food detection.

`best.pt` stores adapted weights, fixed prompt vectors, labels, preprocessing, source revision, and training metadata. It is not a self-contained full-model, ONNX, or TorchScript export. Moving it to another machine also requires the pinned base cache. Back up both separately from Git.

The recorded checkpoint SHA-256 is:

```text
b9089dffbfdf8c682ed28826be143a647a24a675847ca4291bf6f061c34466e4
```

Verify a copy with `python -c "from models.config import sha256; print(sha256('runs/phase4/best.pt'))"`. A newly trained checkpoint has its own identity and measurements.

## Results

The dataset contains 50 photos per dish. A stratified, group-aware split with seed 42 gives **280/64/56** photos (70%/16%/14%, after whole-image allocation).

The recorded full run trained for **11 epochs** and selected **epoch 6**:

- Validation: **61/64 top-1 (95.31%)**, **64/64 top-3**, loss **0.215961**.
- Held-out test: **54/56 top-1 (96.43%)**, **56/56 top-3 (100%)**.
- Macro F1: **0.9641**; macro one-vs-rest ROC-AUC: **0.9985**.
- Independent loading reproduced all 64 saved validation logit rows exactly in the recorded environment.

The two errors were nihari predicted as chicken karahi, and haleem predicted as chapati; both correct labels ranked second. The original evaluator failed after inference during an underscore/space label join; the corrected evaluation used unchanged weights and data. The disclosure is preserved in `evaluation.json` under `provenance.rerun`.

## Files and code

```text
models/
  __init__.py        Python package marker
  config.py          Settings and reading/writing the three evidence files
  data.py            Dataset validation, transforms, and loaders
  clip_classifier.py CLIP architecture and checkpoint saving/loading
  train.py           Training, epoch selection, and checkpoint verification
  evaluate.py        Held-out evaluation
  metrics.py         Classification and ROC calculations
  report.py          Reports and figures from saved predictions
  infer.py           CLI and Recognizer API
evidence/
  training.json
  predictions.csv
  evaluation.json
data/                Source images, manifests, nutrition, generated processed data
data scripts/        Audit, cleaning, preprocessing, split, EDA, reproducibility
docs/                Guides and generated reports
figures/             EDA and model-result figures
runs/                Local checkpoints and pretrained cache (ignored by Git)
scripts/             Contribution audit, weights packaging, dataset verification
```

The model package contains nine Python files, including its package marker. Dataset checks are grouped with data loading, evidence helpers with configuration, checkpoint handling with the classifier, and verification with training. The old phase-specific orchestrator, pilot runner, duplicate verifier, and duplicate report exporter have been removed. Historical experiment details remain in the consolidated provenance records.

## Troubleshooting

- **PowerShell blocks activation:** call `.venv\Scripts\python.exe` instead of `python`; activation is optional.
- **CUDA unavailable:** use the CPU installation and `--device cpu`.
- **Base-model loading fails:** run the pinned download command and retain the full `runs/hub/` cache.
- **Missing `best.pt`:** obtain the evaluated weights or train a new model. Installing dependencies does not create fine-tuned weights.
- **Missing prepared manifests:** run data preparation before training or verification.
- **Existing experiment/evaluation:** choose fresh directories for new work; use saved predictions for reports.
- **Data/checkpoint mismatch:** install requirements.txt (including Pillow 13.0.0), regenerate data, then run python scripts/verify_dataset.py. PNG compression bytes can differ under Pillow 11.3.0 even with identical decoded RGB arrays. Check labels, images, preprocessing and source revision; retain the original agreement checks.
- **`No module named models`:** run `python -m models.<command>` from the repository root.

## Data and limitations

The image sources are four public Roboflow Universe datasets: IndianFoodRecog (EnVisionAI), combined-dataset (Rafias Workspace), Bangladeshi Food Classification2 (NSN), and samosa. Uploaders stated **CC BY 4.0** licensing. [data/sources.csv](data/sources.csv) and [data/metadata.csv](data/metadata.csv) preserve source URLs, attribution, transformations, and recorded hashes. Retain image attribution when reusing them. No repository-wide software license file is currently included.

Source audits verify committed image hashes. Original provider archives/README files are unavailable, so their recorded archive hashes cannot be rechecked. Second human label review remains pending. Only seven test examples per dish, confounded sources/resolutions, and largely Indian/Bangladeshi appearances limit interpretation. External Pakistani meals, mixed plates, and unfamiliar-food rejection remain untested or unsupported; demographic data are unavailable.

Nutrition is separate: `data/nutrition.csv` supplies per-100-g references, and the intended calculation is `value_per_100g * serving_grams / 100` after user confirmation. The inference API returns dish rankings only. Nutrition estimates depend on recipe and serving weight and are not medical advice.

## Team and guides

- **Hassan Mehmood (24L-2559):** technical lead, repository structure, CLIP input/split plan, report integration, and final checks for Deliverable 1.
- **Mohid Israr (24L-2510):** image acquisition, source records, dish definitions, and labels for Deliverable 1.
- **Adnan Zaka (24i-2537):** data cleaning, nutrition references, and exploratory charts for Deliverable 1.

Read the [project overview](docs/01_high_level_explaination.md), [model implementation guide](docs/02_model_implementation.md), and [training/testing guide](docs/03_model_testing.md). The earlier [Deliverable 1 report](Deliverable%201%20Report.pdf) remains available.
