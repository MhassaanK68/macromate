# Macromate: model implementation

The current implementation groups related responsibilities into **nine Python files in `models/`**, including `__init__.py`. The main development workflow is **train -> evaluate -> report**. New-photo inference runs separately.

Read [the overview](01_high_level_explaination.md) first for the project story, and [the training/testing guide](03_model_testing.md) for a complete experiment sequence. All shell commands assume the repository root and the environment described in [the README](../README.md).

## File responsibilities

### `__init__.py`: package marker

This marks `model` as a Python package so commands such as `python -m models.train` can use relative imports. It has no training or inference logic.

### `config.py`: settings and experiment-file helpers

[config.py](../models/config.py) contains the pinned CLIP name/revision, preprocessing constants, training defaults, directory paths, and metric definitions. It also owns JSON/CSV reading and writing for the three evidence files.

Its helpers include `read_labels()`, `preprocessing()`, `sha256()`, `write_json()`, `read_record()`, `reserve_record()`, `write_predictions()`, and `read_predictions()`. `code_hashes()` fingerprints the eight implementation modules; `__init__.py` is only the package marker.

The selected defaults are block LR 2e-5, norm/projection LR 5e-5, batch size 8, weight decay 0.01, maximum 20 epochs, patience 5, seed 42, and dropout zero. Configuration and evidence serialization now share this file.

### `data.py`: validation, preprocessing, and loaders

[data.py](../models/data.py) owns both dataset checks and pixel loading:

- `read_unique()` rejects empty or repeated image IDs.
- `load_records()` joins `clip_manifest.csv` and `split.csv`, checks `label_map.json`, validates paths and label agreement, keeps related groups in one split, and requires 35/8/7 images per dish.
- `data_lineage()` records manifest hashes and training/validation prepared-file hashes.
- `image_tensor()` prepares an ordinary photo or validates an already prepared PNG, then scales and normalizes its pixels once.
- `DishDataset` and `make_loaders()` expose training and validation batches. They reject requests for the reserved test split.

Original photos receive EXIF correction, RGB conversion, bicubic shorter-side resize, and a centered 224 x 224 crop. Prepared inputs must already be RGB 224 x 224 PNGs. Both paths divide values by 255 and apply CLIP channel means/stds. Dataset validation and image loading now share this file.

### `clip_classifier.py`: architecture and checkpoint handling

[clip_classifier.py](../models/clip_classifier.py) implements `DishClassifier`, pretrained loading, and checkpoint saving/reconstruction.

`load_pretrained()` loads `openai/clip-vit-base-patch32` at revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` from the local `runs/hub/` cache. Download that revision before training or inference.

The ViT-B/32 image tower has 12 transformer blocks. Image and text projections each produce 512-dimensional vectors. Eight fixed prompts, `a photo of Pakistani <dish>`, provide normalized text prototypes. The normalized image vector is compared with them using scaled dot products. There is no added classification head.

Only vision blocks 10 and 11, final vision normalization, and visual projection train: **14,570,496 parameters**. Other layers and the text encoder remain frozen.

`save_checkpoint()` writes the adapted weights, text vectors, label order, prompts, preprocessing, base revision, trainable-layer inventory, and selected training metadata. `read_checkpoint()` validates that format; `apply_checkpoint()` checks and copies the saved tensors; `load_checkpoint()` reconstructs the base, applies the adaptations, and switches to evaluation mode.

The weights remain compatible with the existing `best.pt`; grouping checkpoint functions with the classifier does not change parameter names or the checkpoint format. The checkpoint still needs the pinned base cache. ONNX/TorchScript exports are not implemented.

### `train.py`: training and checkpoint verification

[train.py](../models/train.py) owns both `train_run()` and `verify()`.

`train_run()` validates fresh destinations and data, reserves `training.json`, trains using only train/validation loaders, saves improvements to `best.pt`, and records each epoch in the same JSON. It selects the highest validation top-1 count, then lower loss, retaining the earlier epoch on an exact tie. Five consecutive epochs without improvement trigger early stopping.

The objective is unweighted categorical cross-entropy without label smoothing. AdamW excludes biases and normalization from weight decay. A 10% warmup precedes cosine decay, and gradients are clipped at 1.0.

After training, `verify()` reconstructs the saved recognizer and checks checkpoint identity, metadata, data lineage, every selected validation logit, original-photo preprocessing, and the public prediction API. Its result is stored inside `training.json` before training is marked completed. A caught training/verification failure is recorded in the same file.

Verification also has a standalone mode in this module:

```sh
python -m models.train --verify-only --evidence evidence/reproduction --device cpu
```

This checks an existing experiment without running the optimizer or creating another checkpoint. It replaces the need for a separate verification script.

### `evaluate.py`: held-out evaluation

[evaluate.py](../models/evaluate.py) contains `HeldOutDataset`, the evaluation protocol checks, metadata attachment, subgroup summaries, and `evaluate()`.

It reads the selected checkpoint from `training.json`, requires completed training and matching verification/data/model metadata, and validates raw source labels before inference. It exclusively creates `evaluation.json` with the protocol and status `started`, then scores the reserved photos. Successful inference writes `predictions.csv` and completes `evaluation.json` with metrics, subgroups, model details, and error-review slots. A failed attempt remains recorded and blocks an accidental repeat.

The pre-inference protocol includes checkpoint identity, selected epoch, labels, manifest/source/code hashes, test-file hashes, and metric definitions. There are no separate prepare/freeze commands.

### `metrics.py`: numerical measurements

[metrics.py](../models/metrics.py) supplies training/validation summaries, the checkpoint selection key, classification metrics, and binary ROC calculations. Evaluation and reporting reuse these definitions.

It computes top-1/top-3 accuracy, confusion counts, per-class precision/recall/F1, macro/weighted averages, and one-vs-rest ROC-AUC. Tied ROC scores are grouped. AUC is undefined when a class has no positives or negatives. Softmax scores are ranking values, not calibrated confidence.

### `infer.py`: predictions for an ordinary photo

[infer.py](../models/infer.py) exposes `Recognizer(checkpoint, device).predict(photo)` and the CLI. It loads the saved recognizer once, prepares each image, runs without gradients, and returns eight ranked entries containing `label_id`, `dish`, `logit`, and `score`.

Inference imports shared configuration, data transforms, and classifier/checkpoint functions. It does not import `train.py`, read evidence files, or require training images. Reuse one recognizer for multiple photos.

### `report.py`: reports from saved results

[report.py](../models/report.py) reads the three evidence files, checks their identities and prediction integrity, and recomputes metrics/subgroups exactly. It does not load model weights or run inference. The original error photos are needed for their montage.

It writes `docs/results.md`, `docs/model_report.tex`, and learning curves, confusion counts, ROC curves, and all-error figures under `figures/model/`. `--output` and `--figures` select other destinations. Missing visual error observations are labeled pending.

The report generator owns the implementation explanation in generated reports, so rebuilding them retains the current nine-file layout.

## How the components work together

```text
train.py
  -> config.py: settings, paths, evidence reading/writing
  -> data.py: validated records and train/validation batches
  -> clip_classifier.py: pretrained classifier and best.pt
  -> metrics.py: loss/accuracy summaries and epoch selection
  -> verify(): reconstruct checkpoint and compare validation predictions

evaluate.py
  -> training.json: selected checkpoint and successful verification
  -> data.py + clip_classifier.py: test inputs and selected recognizer
  -> metrics.py: recognition measurements
  -> config.py: predictions.csv and evaluation.json

report.py
  -> training.json + predictions.csv + evaluation.json
  -> recomputed metrics, reports, and figures

infer.py
  -> new photo + best.pt + pretrained base cache
  -> eight ranked dishes
```

## Saved files and safeguards

Each experiment uses three evidence files. `training.json` holds settings, environment, labels/prompts, preprocessing, lineage, split counts, history, selected results, validation logits, verification, and checkpoint path/hash. The migrated run also records historical pilot provenance.

`predictions.csv` has one row per test photo, including attribution and complete logits/scores/rankings. List-valued cells use JSON to retain the floating-point values exactly. `evaluation.json` holds the protocol/status, metrics, confusion matrix, subgroup results, model details, and visual error reviews.

JSON updates use atomic replacement. Fresh training rejects existing experiment artifacts or a nonempty weight directory. Exclusive evaluation-record creation prevents repeat attempts. Historical records retain the source identities and hashes that applied when they were produced; source-file consolidation does not rewrite the original experiment history.

Large model files stay in `runs/`: the recorded checkpoint is `runs/phase4/best.pt`, and its required base cache is `runs/hub/`. There are no duplicated per-epoch JSON logs under the active run directory.

The current guides describe the active layout. Historical source identities and experiment context remain in the consolidated records. A historical archive is not present in this checkout and is not required by the model commands.
