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
data/metadata.csv               Image manifest; headers only until acquisition
data/nutrition.csv              Nutrition reference table; headers only until research
data/sources.csv                Dataset/source provenance; headers only until acquisition
data/raw/                       Local raw images (ignored by Git)
data/processed/                 Generated cleaned data (ignored by Git)
figures/                        Report figures after real data is collected
report/main.tex                 LaTeX report draft
report/main.pdf                 Compiled report when complete
```

The internal `planning/` guide is excluded locally and is **not part of the course submission**.

## Current data status and usage conditions

No raw image dataset or nutrition values have been added yet. The header-only CSV files are schemas, **not evidence of data acquisition**. Mohid will record each image source in `data/sources.csv`, including its URL, access date, and license or usage terms. Images will be stored in `data/raw/` locally and will not be uploaded to GitHub unless redistribution is allowed. Adnan will record a source and recipe assumptions for each nutrition row.

To obtain the raw images as a reviewer, follow the source URLs and download/import instructions that will be added to `data/sources.csv` and this README after sources are chosen. This is **not yet reproducible** because the sources have not been selected. If a source restricts sharing, the README will describe its authorized download process rather than copy the files into the repository.

## Set up the current data scripts

Use Python 3.10 or later. In PowerShell from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Add image records to `data/metadata.csv` after placing allowed images under `data/raw/`. `local_path` is relative to the repository root, for example `data/raw/biryani/image_001.jpg`. Use one of the exact dish labels above. Record the image's `source_id` in `data/sources.csv`.

Run the initial cleaning and chart scripts:

```powershell
python "data scripts/clean_data.py"
python "data scripts/eda.py"
```

The cleaning script writes `data/processed/metadata_clean.csv` and `data/processed/cleaning_summary.json` when data is present. It checks required fields, supported labels, readable image files, duplicate IDs and image bytes, and basic nutrition values. The chart script writes `figures/raw_class_counts.png` after at least one metadata row exists. With the current empty CSVs, the scripts report that data collection is pending; they do not fabricate charts or counts. The Phase 2 pipeline and EDA will be expanded once real data is collected.

## Compile the LaTeX report

The report source is a **draft**, not the final Deliverable 1 report. It must be filled with actual collection evidence, cleaning results, and EDA figures before submission. With a LaTeX installation:

```powershell
Set-Location report
pdflatex main.tex
pdflatex main.tex
Set-Location ..
```

The course requires a LaTeX-compiled PDF, a GitHub repository with active participation from all members, raw-data figures, and the Phase 1 and Phase 2 evidence described in the handout. The proposal PDF alone does not satisfy those requirements.
