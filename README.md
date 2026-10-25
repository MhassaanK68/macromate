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

Pakistani dishes vary in recipe and appearance. Fixed rules based on colour or shape are unlikely to recognize them reliably from photos. We will first test a pretrained CLIP model without changing its weights. Then we will fine-tune a small part of CLIP's image encoder on labeled Pakistani food photos, keeping its text encoder fixed. Both versions will be compared on the same held-out test photos using top-1 accuracy, top-3 accuracy, per-dish results, and a confusion matrix. The better-performing version will be considered for the app.

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
report/main.tex                 Completed Deliverable 1 LaTeX technical report
report/preview.py               PDF export and browser preview with automatic rebuilds
report/main.pdf                 PDF produced beside main.tex by report/preview.py
```

The internal `planning/` guide is excluded locally and is **not part of the course submission**.

To export the report to `report/main.pdf`, install a TeX distribution with `pdflatex`, `latexmk`, or `tectonic` on PATH, then run `python report/preview.py --build` from the repository root. Run `python report/preview.py` to open a browser preview that rebuilds when `report/main.tex` or a figure changes; stop it with Ctrl+C. The pip package named `pdflatex` is a wrapper and does not include the compiler executable. The desktop editor compiles its live preview separately and does not write that preview into the repository.

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

The Deliverable 1 report source is complete for the checked-in data. It explicitly records the pending second human label review and unavailable original acquisition archives. With a LaTeX installation:

```powershell
Set-Location report
pdflatex main.tex
pdflatex main.tex
Set-Location ..
```

The course requires a LaTeX-compiled PDF, a GitHub repository with active participation from all members, raw-data figures, and the Phase 1 and Phase 2 evidence described in the handout. The proposal PDF alone does not satisfy those requirements.
