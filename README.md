# Macromate

Macromate is an academic prototype for logging selected Pakistani foods from meal photos. A user uploads a photo of one main dish, reviews the suggested label, enters the serving size in grams, and receives estimated calories and macronutrients. A daily log and simple goal-based messages are planned for later phases.

## Team

| Member | Roll number | Deliverable 1 responsibility |
| --- | --- | --- |
| Hassan Mehmood | 24L-2559 | Technical lead, repository structure, CLIP input and data split plan, LaTeX report integration, final checks |
| Mohid Israr | 24L-2510 | Image acquisition, source records, dish definitions, image labels |
| Adnan Zaka | 24i-2537 | Data cleaning, nutrition references, exploratory charts |

Each member should make substantive commits under their own GitHub account. The current scaffolding commit does not represent their later contributions.

## Problem and AI approach

Pakistani dishes vary in recipe and appearance. Fixed rules based on colour or shape are unlikely to recognize them reliably from photos. We will first test a pretrained CLIP model without changing its weights. Then we will fine-tune a small part of CLIP's image encoder on labeled Pakistani food photos, keeping its text encoder fixed. Both versions will be compared on the same held-out test photos using top-1 accuracy, top-3 accuracy, per-dish results, and a confusion matrix. The better-performing version will be considered for the app.

The initial labels are **chicken biryani, chicken pulao, chicken karahi, daal, nihari, haleem, samosa, and chapati**. This list may change after a data audit. The model suggests a dish; the user confirms or corrects it. Nutrition is a separate calculation based on a documented value per 100 g and the user's entered serving weight. It is an estimate, not medical advice.

## Repository layout

```text
Macromate_Project_Proposal.pdf  Approved project direction
README.md                       Project setup and status
.gitignore                      Excludes raw images and build files
requirements.txt                Dependencies for current data scripts
data scripts/                   Phase 2 Python scripts (folder name required by handout)
data/review_decisions.csv       Manual near-duplicate review decisions (currently none needed)
data/metadata.csv               Image manifest: 400 images, source, license, original file
data/nutrition.csv              Nutrition per 100 g for 8 dishes, with source and recipe assumptions
data/sources.csv                Dataset/source provenance; headers only until acquisition
data/images/                    Local images, one folder per dish (ignored by Git)
data/processed/                 Generated cleaned data (ignored by Git)
figures/                        Report figures after real data is collected
report/main.tex                 LaTeX report draft
report/main.pdf                 Compiled report when complete
```

The internal `planning/` guide is excluded locally and is **not part of the course submission**.

## Current data status and usage conditions

The collection contains **400 images, 50 for each of the eight dishes**, selected from four public Roboflow Universe datasets released under **CC BY 4.0** (as stated by their uploaders): IndianFoodRecog (EnVisionAI), combined-dataset (Rafias Workspace), Bangladeshi Food Classification2 (NSN) and samosa. `data/sources.csv` records each dataset's URL, version URL, access date, license, export format and archive SHA-256. `data/metadata.csv` records, for every image, its source, author, license, original file name inside the dataset archive, and the SHA-256 of the image file. Nutrition values for all eight dishes (per 100 g, with source, calculation and recipe assumption) are in `data/nutrition.csv`.

Images are stored in `data/images/` locally and are **not uploaded to GitHub**. To obtain them as a reviewer, open each dataset's `version_url` in `data/sources.csv`, download it in the listed export format, check the archive hash, and copy the files named in the `original_path` column of `data/metadata.csv` to the `local_path` given there. Credit the original authors when reusing images.

Known limits: the second human review of labels is still pending for all images; the sources are largely Indian and Bangladeshi, so Pakistani appearance is only partly represented; the providers resized every image to a square, so image size depends on the source (see the report's EDA section).

## Set up the current data scripts

Use Python 3.10 or later. In PowerShell from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Add image records to `data/metadata.csv` after placing allowed images under `data/images/` (git-ignored). `local_path` is relative to the repository root, for example `data/images/daal/RF_COMBINED_0123456789abcdef.jpg`. Use one of the exact dish labels above. Record the image's `source_id` in `data/sources.csv`.

Optional metadata column: add `group_id` to `data/metadata.csv` and give photos of the same physical plate the same value, so the split keeps them together.

Run the Phase 2 pipeline in this order (from the repository root):

```powershell
python "data scripts/clean_data.py"     # 4.1 validate, 4.2 duplicates + outliers + nutrition checks
python "data scripts/preprocess.py"     # 4.3 CLIP preprocessing + dish-name -> integer ID mapping
python "data scripts/split_data.py"     # 4.4 seeded, stratified, leakage-aware train/val/test split
python "data scripts/eda.py"            # EDA charts in figures/
python "data scripts/verify_reproducible.py"   # 4.5 runs the pipeline twice and compares outputs
```

| Script | What it does | Main outputs (`data/processed/`) |
| --- | --- | --- |
| `common.py` | Dish list, label IDs, CLIP constants, split seed | (shared by all scripts) |
| `clean_data.py` | Checks columns, labels, sources, readable files; removes exact duplicates (and copies under conflicting labels); lists near-duplicate pairs by perceptual hash for **human review**; flags image outliers (not removed); validates nutrition values | `metadata_clean.csv`, `exclusions.csv`, `near_duplicate_pairs.csv`, `outlier_flags.csv`, `nutrition_clean.csv`, `cleaning_summary.json` |
| `preprocess.py` | Official CLIP image preprocessing (RGB, resize 224, centre crop, normalise); stable label IDs; nutrition kept per 100 g | `clip_224/`, `label_map.json`, `preprocess_summary.json` |
| `split_data.py` | 70/15/15 split per dish with seed 42; near-duplicates and same-plate photos stay in one split; runs leakage checks | `split.csv`, `split_summary.json` |
| `eda.py` | Class counts, missing data, sample photos, size distributions, correlations, source mix, split, nutrition charts | `figures/*.png` |
| `verify_reproducible.py` | Runs the pipeline twice, compares SHA-256 of every output | `reproducibility_check.json` |

Near-duplicates are **flagged, never deleted automatically**. After reviewing `near_duplicate_pairs.csv`, record each removal in `data/review_decisions.csv` (`image_id,decision,reason` with `decision` = `drop`) and re-run the pipeline. Outlier flags are for review only. With the current empty CSVs the scripts report that data collection is pending; they do not fabricate charts or counts.

## Compile the LaTeX report

The report source is a **draft**, not the final Deliverable 1 report. It must be filled with actual collection evidence, cleaning results, and EDA figures before submission. With a LaTeX installation:

```powershell
Set-Location report
pdflatex main.tex
pdflatex main.tex
Set-Location ..
```

The course requires a LaTeX-compiled PDF, a GitHub repository with active participation from all members, raw-data figures, and the Phase 1 and Phase 2 evidence described in the handout. The proposal PDF alone does not satisfy those requirements.
