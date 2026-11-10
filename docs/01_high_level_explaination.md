# Macromate: project overview

Macromate recognizes one of eight supported dish categories from a photo. It is an academic model prototype intended for a meal-logging application.

This guide explains the current nine-file model package. Continue with [the implementation guide](02_model_implementation.md) for file/function details and [the training and testing guide](03_model_testing.md) for commands and results. Laptop setup is in [the README](../README.md).

## The complete story for a presentation

1. Collect and document food images.
2. Clean them and prepare 224 x 224 RGB inputs.
3. Divide them into training, validation, and test sets.
4. Adapt pretrained CLIP using training images.
5. Select and save the best checkpoint using validation images.
6. Reconstruct the saved checkpoint and verify its predictions.
7. Measure it on held-out test images and save the results.
8. Load the checkpoint to suggest dishes for a new photo.

The model output is a dish ranking. The intended app asks the user to confirm the dish and enter grams, then scales a separate nutrition reference. It does not calculate grams or nutrition directly from pixels. The user interface, serving-size workflow, nutrition integration, and daily log remain future work.

## What CLIP does

CLIP represents images and text in the same numerical space. We describe each dish using `a photo of Pakistani <dish>`. The model compares a photo's representation with those eight descriptions. The largest similarity becomes the first suggestion.

We use pretrained CLIP ViT-B/32 at a pinned revision. Fine-tuning adjusts only its final two vision transformer blocks, final vision normalization, and visual projection. Earlier vision layers and the text encoder stay frozen. This limits the weights learned from our small dataset.

The eight labels are chicken biryani, chicken pulao, chicken karahi, daal, nihari, haleem, samosa, and chapati, in class-ID order 0 through 7.

## Where the code lives

The `models/` package has nine Python files, including its package marker:

```text
models/
  __init__.py         Marks the Python package
  config.py           Settings and evidence-file reading/writing
  data.py             Dataset checks, image preparation, and loaders
  clip_classifier.py  CLIP architecture and checkpoint saving/loading
  train.py            Training, epoch selection, and checkpoint verification
  evaluate.py         Held-out testing and saved evaluation results
  metrics.py          Accuracy, precision, recall, F1, and ROC calculations
  infer.py            Dish predictions for a new photo
  report.py           Reports and figures from saved results
```

Related work is grouped together. Dataset validation belongs with data loading; evidence helpers belong with configuration; checkpoint handling belongs with the classifier; verification belongs with training. The architecture and saved model format are unchanged by this grouping.

## Why there are three dataset subsets

There are 400 photos, 50 per dish:

- **Training: 280.** These update the model's weights.
- **Validation: 64.** These select the best epoch and control early stopping.
- **Test: 56.** These measure the selected model after training.

The split is seeded, stratified, and group-aware. Related versions of a source photo stay together so they cannot appear on both sides of a split and make evaluation artificially easy. Whole-image allocation gives a 70%/16%/14% split from the approximate 70/15/15 target.

## What the folders contain

**`models/` contains executable Python code.** It describes how to train, test, save, and run the recognizer.

**`runs/` contains the local model files.** The recorded fine-tuned checkpoint is `runs/phase4/best.pt`, and the pinned pretrained CLIP cache is `runs/hub/`. The checkpoint stores adapted weights and needs the base cache to reconstruct the recognizer. Large weights and the cache are excluded from Git.

**`evidence/` contains three readable experiment records:**

- `training.json`: settings, epoch history, selected checkpoint, validation predictions, and verification.
- `predictions.csv`: every held-out test prediction, scores, rankings, and attribution.
- `evaluation.json`: evaluation protocol/status, metrics, confusion matrix, subgroups, error reviews, and model details.

**`data/` contains source images and manifests.** Prepared images, splits, and source-audit summaries are generated under `data/processed/`. Nutrition references are separate from recognition.

**`docs/` contains guides and generated reports.** The consolidated evidence records retain historical source identities, pilot selection, and the evaluation correction under provenance.

## The commands to present

After installing dependencies, downloading the pinned base, and preparing data, a new experiment follows this sequence from the repository root:

```sh
python -m models.train --output runs/reproduction --evidence evidence/reproduction --device cpu
python -m models.evaluate --evidence evidence/reproduction --device cpu
python -m models.report --evidence evidence/reproduction
```

Choose unused experiment directories. Training includes checkpoint verification automatically. To repeat only the validation/checkpoint check in a separate process, use:

```sh
python -m models.train --verify-only --evidence evidence/reproduction --device cpu
```

Standalone verification uses the same `train.py` file and does not retrain. Test inference is separate and refuses an existing evaluation attempt. Use saved predictions to regenerate reports.

For an ordinary photo and the existing checkpoint:

```sh
python -m models.infer runs/phase4/best.pt "path/to/meal.jpg" --device cpu
```

Replace the photo path with a real image. Inference needs the checkpoint and base cache; it does not read the evidence folder or need the training dataset.

## What the result means

The recorded checkpoint achieved 54/56 top-1 accuracy (96.43%), 56/56 top-3, macro F1 0.9641, and macro ROC-AUC 0.9985. These results apply to a small test set, with seven examples per dish. The two incorrect first suggestions had the correct dish second.

The scores are not calibrated confidence. The recognizer does not reliably reject unfamiliar foods or support mixed plates. Labels still need a second human review, and performance on a separate Pakistani meal collection remains untested.

The module consolidation preserved the original checkpoint and measured test results. The saved evaluation is in `evidence/evaluation.json`; run `python -m models.report` to generate a formatted report when needed. Historical source names in evidence describe the original experiment; the active workflow uses the nine files listed above.
