# Macromate

Macromate is an academic prototype for logging selected Pakistani foods from meal photos. A user uploads a photo of one main dish, reviews the suggested label, enters the serving size in grams, and receives estimated calories and macronutrients. A daily log and simple goal-based messages are planned for later phases.

## Team

| Member | Roll number | Deliverable 1 responsibility |
| --- | --- | --- |
| Hassan Mehmood | 24L-2559 | Technical lead, repository structure, CLIP input and data split plan, LaTeX report integration, final checks |
| Mohid Israr | 24L-2510 | Image acquisition, source records, dish definitions, image labels |
| Adnan Zaka | 24i-2537 | Data cleaning, nutrition references, exploratory charts |

Each member should make substantive commits under their own GitHub account. The repository history now includes commits from all three members.

## Problem and AI approach

Pakistani dishes vary in recipe and appearance. Fixed rules based on colour or shape are unlikely to recognize them reliably from photos. The Phase 3 engine partially fine-tunes CLIP's image encoder while keeping its text encoder fixed. Phase 4 will fully train the selected configuration and evaluate this single recognizer on reserved test photos, including top-1/top-3 accuracy, precision, recall, F1, ROC-AUC and error analysis. The earlier proposal's two-model comparison has been superseded by the Phase 3 and Phase 4 implementation documents.

The initial labels are **chicken biryani, chicken pulao, chicken karahi, daal, nihari, haleem, samosa, and chapati**. This list may change after a data audit. The model suggests a dish; the user confirms or corrects it. Nutrition is a separate calculation based on a documented value per 100 g and the user's entered serving weight. It is an estimate, not medical advice.

## Repository layout

```text
Macromate_Project_Proposal.pdf  Approved project direction
README.md                       Project setup and status
.gitignore                      Excludes generated data, environments and build files
requirements.txt                Dependencies for current data scripts
data scripts/                   Phase 2 Python scripts (folder name required by handout)
data/review_decisions.csv       Manual near-duplicate review decisions (currently none needed)
evidence/                       Repository-derived source audit summaries
data/metadata.csv               Image manifest: 400 images, source, license, original file
data/nutrition.csv              Nutrition per 100 g for 8 dishes, with source and recipe assumptions
data/sources.csv                Dataset/source provenance for four sources
data/images/                    400 committed images, one folder per dish
data/processed/                 Generated cleaned data (ignored by Git)
figures/                        Report figures generated from the current data
report/main.tex                 Data preparation and Phase 3 technical report
report/phase3.tex               Phase 3 section exported from measured pilot evidence
report/main.pdf                 Compiled technical report
model/                          CLIP recognizer, trainer, pilot and inference modules
requirements-model.txt          Tested model dependencies
tests/                          Data and model contract tests
runs/                           Local model cache, weights and histories (ignored)
```

## Phase 3 model engine

**Phase 3 is complete:** all 16 contract tests pass, all six two-epoch GPU pilots
completed, and independent checkpoint loading reproduced all 64 validation logit
rows exactly. The selected candidate reached **58/64 validation top-1** and
**63/64 top-3**, with loss **0.282456**. These are exploratory pilot results;
the 56 test photos remain reserved for Phase 4. See `docs/05_Phase_3_results.md`
and `evidence/phase3/` for measured settings, histories and verification.

Phase 3 uses PyTorch and Hugging Face Transformers with the pinned
`openai/clip-vit-base-patch32` revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. The execution design is in
`docs/04_Phase_3_execution_plan.md`. The eight class IDs come directly from
`data/processed/label_map.json`.

The image tower is ViT-B/32 (12 blocks, width 768, 12 heads). The fixed text
tower has 12 blocks, width 512 and 8 heads; both project to 512 dimensions.
Only vision blocks 10 and 11, final vision layer normalization and visual
projection train. Eight fixed `a photo of Pakistani [dish]` embeddings supply
the classifier scores; no new classification head is added. Most pretrained
features stay frozen because only 280 training photos are available.

Install Python 3.12, create/activate `.venv`, then install the pinned libraries:

```powershell
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-model.txt
# For a CPU-only environment, use https://download.pytorch.org/whl/cpu instead.
```

Download the pinned base model once. It is cached inside ignored `runs/hub/`;
model loading then works offline and uses only this revision:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/clip-vit-base-patch32', revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268', cache_dir='runs/hub', allow_patterns=['*.json','*.txt','pytorch_model.bin'])"
python -m unittest discover -s tests -v
python -m model.pilot --epochs 2 --device cuda
python -m model.verify --device cuda
python -m model.report
```

Use `--device cpu` when CUDA is unavailable. The six declared pilot candidates
change one parameter at a time, each with seed 42 and the same epoch budget.
The objective is unweighted categorical cross-entropy without label smoothing.
AdamW uses separate block and norm/projection learning rates, excludes biases
and normalization weights from decay, and uses 10% warmup, cosine decay and
gradient clipping at 1.0. Pilot datasets expose only train and validation.

Prepared 224x224 RGB PNGs are scaled and normalized once without a second crop
or resize. User uploads use the exact Phase 2 EXIF/resize/crop recipe followed
by that same normalization. `model/data.py` validates both manifests, label IDs,
counts, file paths and group separation before loading pixels.

Pilot weights and full run files live in `runs/phase3/<run_id>/`. Small measured
results, histories, environment details, trainable names, tuning CSV and the
selected configuration are exported to `evidence/phase3/`. A run directory is
never overwritten; completed candidates can be reused only with matching
protocol and data. Failed attempts are recorded as failures, not measurements.
`python -m model.report` exports `report/phase3.tex` and
`docs/05_Phase_3_results.md` only after all six pilots and independent checkpoint
verification succeed. The generated section is included in `report/main.tex`.

The selected configuration uses block LR `2e-5`, norm/projection LR `5e-5`,
batch size 8 and weight decay `0.01`. Highest validation top-1 count determines
selection, then lower loss; the short pilot does not establish optimality.
The pilot checkpoint is `runs/phase3/block_high/best.pt`.

On this machine, the standalone Python 3.12.10 interpreter at
`runs/python312/python.exe` successfully ran the tests and GPU pilots. Substitute
`.\runs\python312\python.exe` for `python` in these commands when using this
local setup. A normal Python 3.12 virtual environment remains the portable setup;
the local standalone interpreter and base model cache are ignored by Git.

To check manifests without loading PyTorch:

```powershell
python -m unittest discover -s tests -p test_data_contracts.py -v
```

Phase 4 starts a fresh pretrained model with the selected configuration, up to
20 epochs and early stopping after five validation epochs without improvement:

```powershell
python -m model.train --config evidence/phase3/selected_config.json --epochs 20 --output runs/phase4 --device cuda
python -m model.infer runs/phase4/best.pt path/to/photo.jpg --device cuda
```

For a Phase 3 inference check, substitute the selected pilot's `best.pt`.
The Python API is `Recognizer(checkpoint, device).predict(photo)`, imported from
`model.infer`. It returns `ranked_dishes` with `label_id`, `dish`, `logit` and
`score`; `scores_are_calibrated` is false. Scores rank only these eight dishes,
and do not measure reliable confidence or detect unfamiliar meals.

Checkpoints store adapted weights, fixed prompt vectors, class order, prompts,
preprocessing, exact source revision, trainable-layer inventory, manifest and
train/validation pixel hashes, settings, environment and selected epoch metrics.
Every successful run verifies saved/reloaded validation logits within 1e-6.
Full training and all held-out test evaluation belong to Phase 4.

The internal `planning/` guide is excluded locally and is **not part of the course submission**.

To rebuild `report/main.pdf`, install a TeX distribution with `pdflatex` on PATH
and use the commands in **Compile the LaTeX report** below. The pip package
named `pdflatex` is a wrapper and does not include the compiler executable.

## Current data status and usage conditions

The collection contains **400 images, 50 for each of the eight dishes**, selected from four public Roboflow Universe datasets released under **CC BY 4.0** (as stated by their uploaders): IndianFoodRecog (EnVisionAI), combined-dataset (Rafias Workspace), Bangladeshi Food Classification2 (NSN) and samosa. `data/sources.csv` records each dataset's URL, version URL, access date, license, export format and archive SHA-256. `data/metadata.csv` records, for every image, its source, author, license, original file name inside the dataset archive, and the SHA-256 of the image file. Nutrition values for all eight dishes (per 100 g, with source, calculation and recipe assumption) are in `data/nutrition.csv`.

The 400 selected images are committed under `data/images/`. Credit the original authors when reusing them. `python "data scripts/audit_sources.py"` checks every committed image against the SHA-256 in `data/metadata.csv` and writes four summaries under `evidence/`. Those summaries are derived from the committed manifest and files. The original Roboflow export archives and provider README files are unavailable, so the archive hashes recorded in `data/sources.csv` cannot currently be rechecked.

Known limits: the second human review of labels is still pending for all images; the sources are largely Indian and Bangladeshi, so Pakistani appearance is only partly represented; the providers resized every image to a square, so image size depends on the source (see the report's EDA section).

## Set up the current data scripts

Use Python 3.10 or later. In PowerShell from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Add image records to `data/metadata.csv` after placing allowed images under `data/images/`. Images in that directory are tracked by Git. `local_path` is relative to the repository root, for example `data/images/daal/RF_COMBINED_0123456789abcdef.jpg`. Use one of the exact dish labels above. Record the image's `source_id` in `data/sources.csv`, and record the image file's SHA-256 in `metadata.csv`.

Optional metadata column: add `group_id` to `data/metadata.csv` and give photos of the same physical plate the same value, so the split keeps them together.

Run the Phase 2 pipeline in this order (from the repository root):

```powershell
python "data scripts/audit_sources.py"  # checks source counts and image hashes; writes derived summaries
python "data scripts/clean_data.py"     # 4.1 validate, 4.2 duplicates + outliers + nutrition checks
python "data scripts/preprocess.py"     # 4.3 CLIP preprocessing + dish-name -> integer ID mapping
python "data scripts/split_data.py"     # 4.4 seeded, stratified, leakage-aware train/val/test split
python "data scripts/eda.py"            # EDA charts in figures/
python "data scripts/verify_reproducible.py"   # 4.5 reruns all five steps and compares outputs
python -m unittest discover -s tests -v       # edge-case regression tests
```

| Script | What it does | Main outputs (`data/processed/`) |
| --- | --- | --- |
| `common.py` | Dish list, label IDs, CLIP constants, split seed | (shared by all scripts) |
| `audit_sources.py` | Verifies committed image hashes and per-source counts; records limits of available provenance | `evidence/*_summary.json` |
| `clean_data.py` | Checks columns, labels, sources, readable files and recorded image hashes; removes exact duplicates (and copies under conflicting labels); lists near-duplicate pairs by perceptual hash for **human review**; flags image outliers (not removed); validates nutrition values | `metadata_clean.csv`, `exclusions.csv`, `near_duplicate_pairs.csv`, `outlier_flags.csv`, `nutrition_clean.csv`, `cleaning_summary.json` |
| `preprocess.py` | Produces 224-pixel RGB PNGs for inspection, records CLIP-normalised pixel statistics, maps each PNG to a stable label ID; nutrition stays per 100 g. Apply CLIP's mean/std normalisation when loading PNGs for a model. | `clip_224/`, `clip_manifest.csv`, `label_map.json`, `preprocess_summary.json` |
| `split_data.py` | 70/15/15 split per dish with seed 42; near-duplicates and same-plate photos stay in one split; runs leakage checks | `split.csv`, `split_summary.json` |
| `eda.py` | Class counts, missing data, sample photos, size distributions, correlations, source mix, split, nutrition charts | `figures/*.png` |
| `verify_reproducible.py` | Runs all five steps twice, checks output completeness and compares SHA-256 of the expected CSV, JSON and PNG outputs | `reproducibility_check.json` |

Near-duplicates are **flagged, never deleted automatically**. After reviewing `near_duplicate_pairs.csv`, record each removal in `data/review_decisions.csv` (`image_id,decision,reason` with `decision` = `drop`) and re-run the pipeline. Outlier flags are for review only. The current dataset has no near-duplicate pairs and four size outlier flags.

## Compile the LaTeX report

The report contains the checked-in data preparation and the measured Phase 3 implementation/pilot. It records the pending second human label review and unavailable original acquisition archives. With a LaTeX installation:

```powershell
Set-Location report
pdflatex main.tex
pdflatex main.tex
Set-Location ..
```

The course requires a LaTeX-compiled PDF, a GitHub repository with active participation from all members, raw-data figures, and the Phase 1 and Phase 2 evidence described in the handout. The proposal PDF alone does not satisfy those requirements.
