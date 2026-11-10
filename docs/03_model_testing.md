# Macromate: training and testing

The model lifecycle is **train -> evaluate -> report**. The current `models/` package has nine Python files, including its package marker. Training and checkpoint verification both live in `train.py`; `clip_classifier.py` owns checkpoint saving/loading; `data.py` owns dataset checks and transforms; `config.py` owns settings and the three evidence-file helpers.

This guide describes the current commands, what each stage reads/writes, and how to interpret the saved result. Read [the overview](01_high_level_explaination.md) for a presentation outline and [the implementation guide](02_model_implementation.md) for all nine file responsibilities.

## Start a new experiment

First follow [the README setup](../README.md): activate the Python environment, install dependencies, download the pinned CLIP base to `runs/hub/`, and run the data preparation scripts. Run model commands from the repository root. Finish data regeneration before starting training because verification checks the exact prepared-file hashes.

Use fresh destinations for a new experiment:

```sh
python -m models.train --output runs/reproduction --evidence evidence/reproduction --device cpu
```

The command saves weights to `runs/reproduction/best.pt` and records training and automatic verification in `evidence/reproduction/training.json`. That JSON supplies the checkpoint path to later commands. The existing evaluated checkpoint and evidence stay at `runs/phase4/best.pt` and the root `evidence/` folder.

Use `--device cuda` when supported. `--epochs 1` provides a short execution check; `--config path/to/settings.json` supplies alternative settings. These define separate experiments. The recorded settings use the 20-epoch cap and patience five.

## Training and validation

The 400-image dataset has eight dishes and 50 images per dish. The fixed, stratified, group-aware split assigns 280 images to training, 64 to validation, and 56 to testing. Each dish has 35/8/7 examples respectively.

Training images update weights. Validation images are measured after each epoch, without gradient updates. The best checkpoint has the greatest validation top-1 correct count, with lower loss breaking ties; exact ties keep the earlier epoch. Training stops after five epochs without improvement or the 20-epoch cap.

The recorded full run completed 11 epochs and selected epoch 6:

- Validation top-1: **61/64 (95.31%)**.
- Validation top-3: **64/64 (100%)**.
- Validation cross-entropy: **0.215961**.

Training started from the pinned pretrained CLIP base, using settings selected by six earlier short pilot experiments. It did not continue a pilot's weights. Those pilot measurements now live in `training.json` under `provenance`. New runs use the selected defaults directly.

## Checkpoint verification

After training, a recognizer is reconstructed from `best.pt`. All 64 selected validation logits must reproduce within absolute tolerance `1e-6` on the same device, or `1e-4` across devices. The verifier also checks the checkpoint identity, data lineage, original-photo preprocessing, and public inference API.

The `verify()` function in `models/train.py` performs this check automatically after training. An optional independent process check uses the same module:

```sh
python -m models.train --verify-only --evidence evidence/reproduction --device cpu
```

The `--verify-only` mode does not retrain, update weights, or run held-out test inference. It updates verification inside the same `training.json` and produces no additional file. The original recorded verification matched all validation logits exactly, with maximum error 0.0.

## Held-out evaluation

After training and verification complete, run:

```sh
python -m models.evaluate --evidence evidence/reproduction --device cpu
```

The evaluator reads the selected checkpoint path from `training.json`. It checks successful training/verification, checkpoint metadata and hashes, label order, data lineage, and source-metadata joins. It then creates `evaluation.json` exclusively with status `started` and records the protocol before loading test pixels.

A successful pass writes all predictions to `predictions.csv` and completes `evaluation.json` with metrics, subgroups, model details, and error-review slots. A failure records its error in the same JSON. Both started and failed records prevent accidental repeat inference. A race between evaluators is also blocked by exclusive file creation.

There are no separate prepare or freeze commands. A genuine bug-fix rerun must preserve and disclose the earlier attempt. Once test outcomes inform model changes, the same test images do not provide a fresh unbiased evaluation.

## Recorded test results

The current consolidated files preserve the original corrected evaluation of the unchanged checkpoint:

- Top-1: **54/56 (96.43%)**.
- Top-3: **56/56 (100%)**.
- Macro F1: **0.9641**.
- Macro one-vs-rest ROC-AUC: **0.9985**.

The first evaluator completed inference but failed while joining raw labels containing underscores with prepared labels containing spaces. Its correction applied the existing normalization and validated metadata before inference. Model settings, weights, labels, and split were unchanged. No prior outcomes were used for tuning. The disclosure is in `evaluation.json` under `provenance.rerun`. The historical archive and original phase-specific attempt files are absent from the current checkout; use the retained provenance to understand the correction.

The saved evaluation was migrated without retraining its checkpoint or repeating held-out inference. All 56 full-precision prediction records were converted losslessly to the CSV. Separate validation runs used during the module refactor reproduced the original trained tensors and validation logits; they did not replace the evaluated checkpoint or create new held-out performance claims.

## Interpreting metrics

Top-1 measures the first suggestion; top-3 measures whether the real label appears among three alternatives. Precision measures correctness when a dish is suggested. Recall measures how often a real dish is found. F1 balances precision and recall. Macro averages weight dishes equally; weighted averages use true class support.

ROC-AUC measures one-vs-rest ranking across thresholds. Equal scores are grouped. AUC is undefined if a class has no positive or no negative examples. High AUC does not establish calibrated probabilities. Confusion-matrix rows are true labels and columns predicted labels.

The two recorded errors were nihari -> chicken karahi and haleem -> chapati. Both true dishes ranked second. The visual observations are in `evaluation.json` under `error_reviews`. Shared meat-curry appearance and flat brown shape/context are possible cues, not proven causes.

## Reports and review

To generate reports from the existing saved analysis:

```sh
python -m models.report
```

For the new experiment:

```sh
python -m models.report --evidence evidence/reproduction
```

The report generator checks the CSV hash, model identity, test coverage, score/ranking consistency, and exact metric/subgroup reproduction. It needs the three evidence files, plus original error photos for their montage. It does not require the model weights or prepared training dataset.

The command creates `docs/results.md`, `docs/model_report.tex`, and `figures/model/`. Generated report sources may be absent until the command is run. See the README for optional PDF compilation and custom output paths. Review new errors and write observations in `evaluation.json`; otherwise the report labels the visual review pending.

## Run the saved model

For the recorded checkpoint:

```sh
python -m models.infer runs/phase4/best.pt "path/to/meal.jpg" --device cpu
```

For your reproduction, use `runs/reproduction/best.pt` instead. Replace the image path with a real photo and retain the pinned base cache in `runs/hub/`. Inference reads the checkpoint using `clip_classifier.py` and transforms the photo using `data.py`; it does not read the evidence files or import training.

The returned JSON contains eight ranked dishes with IDs, names, logits, and softmax scores. The intended app asks the user to confirm or correct the dish before using a nutrition reference.

## Software tests and limitations

Run `python -m unittest discover -s tests -v` for the current 19-test regression suite. These software checks cover serialization, metrics, failure handling, saved-result fidelity, artifact validation, repeat-evaluation protection, and the consolidated workflow. A dedicated command test checks that `--verify-only` never starts training. Evaluator integration tests use numerical synthetic inputs to exercise the output files without repeating real held-out-photo inference. They are distinct from recognition accuracy. Historical quality-check logs are retained under evaluation provenance and are distinct from newly executed checks. The saved-prediction fidelity test reconstructs the original JSON bytes from the CSV and compares them with the retained original SHA-256. It accounts for the original Windows CRLF line endings. Historical checks also verify that the evaluation-correction disclosure remains in provenance. These checks work without the removed archive.

The test set has only seven photos per dish. Sources have different dish mixtures and provider resolutions, so source/dish/size effects are confounded. External Pakistani meal performance is untested. Second human label review is pending, demographic data are unavailable, and scores are uncalibrated. The model does not reliably reject unfamiliar foods or support mixed plates.

## What to say in a presentation

The nine source files implement the behavior; the three evidence files explain the run; the checkpoint and pretrained cache contain the model weights. Moving related functions into shared modules changes their location, while preserving their checks and numerical behavior.

For a presentation, explain: **we train using 280 photos, select with 64 validation photos, save the best model, and measure it on 56 reserved photos. Three files tell us how it trained, what it predicted, and how well it performed.**
