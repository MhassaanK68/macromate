# Phase 4 implementation plan: Training & Evaluation

Phase 4 uses the **single CLIP recognizer implemented in Phase 3**. We will train it with the selected, documented configuration; choose the checkpoint using validation data; and assess the locked model on reserved test photos. The required report will explain not only overall accuracy but also precision, recall, F1-score, multiclass ROC-AUC, errors, and possible bias. This is an evaluation of the model intended for the app, not a comparison with another model.

The Phase 2 split targeted approximately 70/15/15 and is group-aware. Its actual counts are **280 train, 64 validation and 56 test** (35/8/7 per dish), or 70%/16%/14% after integer allocation. Retain the existing assignments. The seven test photos per dish limit how precise any result can be.

## Steps to complete Phase 4

- **Step 1 — Verify the selected configuration and validation strategy.**
- **Step 2 — Run the full training and select the best checkpoint.**
- **Step 3 — Freeze the evaluation protocol and run the held-out test.**
- **Step 4 — Calculate and explain the required classification metrics.**
- **Step 5 — Analyze mistakes and class-level behavior.**
- **Step 6 — Perform a bias and data-source check.**
- **Step 7 — Write the results, limitations and model handoff.**

## Detailed Plan for Each Step

### Step 1: Verify the selected configuration and validation strategy

- Read the Phase 3 tuning record and selected settings. Confirm the checkpoint revision, prompt strings, layer-freezing policy, optimizer, learning rates, batch size, weight decay, epoch cap, early-stopping rule, random seed and preprocessing. Record the hashes of the Phase 2 manifests and split so results can be traced to the precise data used.
- Use the existing **train/validation/test split** as the course brief's required validation technique. Training photos update weights; validation photos guide epoch selection and any Phase 3 parameter choice; test photos are reserved for the final locked evaluation. The split is stratified by dish and keeps related source-photo groups together. K-fold cross-validation is an alternative, not an additional requirement for this dataset, and would complicate keeping the reserved test set untouched.
- Check each subset has all eight labels and that no `image_id` or group crosses subsets. Do not regenerate the split to seek a better outcome. If the data changed since Phase 3, rerun the Phase 2 pipeline and explicitly treat the result as a new experiment.
- **Step is complete when:** the configuration and data lineage match Phase 3, the split integrity checks pass, and the validation strategy is stated in the report.

### Step 2: Run the full training and select the best checkpoint

- Train the Phase 3 architecture on the 280 training photos using the selected hyperparameters. Run up to the planned epoch limit with the stated early-stopping rule. Log training loss, validation loss, validation top-1 and top-3 accuracy, learning rate, epoch duration and any numerical errors. Record the actual device, precision and software versions.
- After each epoch, score the 64 validation photos without updating weights. Select the checkpoint by highest validation top-1 correct count; break ties with lower validation cross-entropy, then earlier epoch. Keep per-class validation counts so an apparent overall gain can be inspected for uneven performance.
- Reload the selected checkpoint and confirm it reproduces the saved validation predictions. Persist weights, prompts, label order, normalization constants, source revision, split hashes, selected epoch and run history. Do not use the test set to choose an epoch or revise hyperparameters.
- **Step is complete when:** one selected, reloadable fine-tuned checkpoint and its complete training/validation history are available.

### Step 3: Freeze the evaluation protocol and run the held-out test

- Before reading test predictions, write down the exact metric definitions, class order, averaging methods and reporting format. Use the single selected checkpoint from Step 2, deterministic preprocessing, and all 56 reserved test images exactly once for the final result. Save each image ID, true label, ranked scores and predicted label for audit and error analysis.
- A later bug fix can justify rerunning the evaluation, but document the code change and prior result. Do not tune the model or prompts in response to test mistakes and then report the same test set as a fresh unbiased assessment.
- Use `model/evaluate.py` for the test run and `model/metrics.py` for calculations. Add `scikit-learn` as a tested metrics dependency if using its precision, recall, F1, confusion-matrix and ROC-AUC functions. Use the exact score ordering from the saved label map.
- **Step is complete when:** the locked model has one traceable set of per-image test predictions and the evaluation settings are recorded.

### Step 4: Calculate and explain the required classification metrics

- Report **top-1 accuracy** as correct first-choice predictions divided by 56 and **top-3 accuracy** as photos whose correct class appears among the first three. Include raw correct/total counts. These tell us how often the app's suggestion is right and how often the right dish is among alternatives.
- For each of the eight dishes, report **precision** (of photos predicted as that dish, how many truly are it), **recall** (of true photos of that dish, how many the model found), and **F1-score** (a balance of precision and recall). Include support, the number of true test photos for that dish. If the model never predicts a dish, define its precision as zero and disclose that convention.
- Report **macro averages** so each dish contributes equally, plus **weighted averages** if support changes in a future dataset. Because this test set has seven images per dish, macro and weighted averages should be close or identical under standard support weighting. Accuracy alone could hide a model that repeatedly misses a particular dish; these class-aware metrics reveal that failure.
- For the brief's **AUC-ROC** requirement, use multiclass **one-vs-rest**: for each dish, treat it as positive and the other seven as negative, using the model's continuous class score or softmax output for ranking. Report each class's ROC-AUC and their macro average; optionally show one-vs-rest ROC curves. ROC-AUC describes ranking across thresholds and does not prove that softmax values are calibrated probabilities. Each class has only seven positives, so curves and AUC values will be noisy. State this limitation and do not force a value if a class lacks positives or negatives after a future data change.
- Save the exact metric definitions, raw counts, per-class results and figure inputs. Do not infer a meaningful percentage-point improvement from a single image on this small test set.
- **Step is complete when:** accuracy, precision, recall, F1 and multiclass ROC-AUC are reproducibly calculated and their relevance and limitations are explained in the report.

### Step 5: Analyze mistakes and class-level behavior

- Produce an eight-by-eight **confusion matrix** with true dishes on one axis and predicted dishes on the other. Show counts, not just colors. Inspect every incorrect test image by ID, true/predicted dish, rank of the true dish, source and visible context. Pay special attention to chicken biryani versus chicken pulao and other visually related foods.
- Distinguish plausible visual confusion, questionable labels, cropping problems, and source-specific artifacts where evidence allows. Do not silently relabel or remove hard test examples after seeing the results; corrections must be documented and any revised evaluation identified as such.
- Describe the app consequence of each important error. For example, a wrong dish suggestion can lead to wrong nutrition values unless the user corrects it. Keep correction simple and avoid presenting a score as certainty.
- **Step is complete when:** the report identifies the main error patterns with supporting counts and representative, attributed examples where use rights permit.

### Step 6: Perform a bias and data-source check

- Include a report section explicitly titled **Bias Check**. Inspect test outcomes by `source_id`, original dataset, dish, image size or preprocessing style where sample counts allow. Report group sizes and correct/total counts alongside percentages. The source datasets include Indian and Bangladeshi foods, and some dishes occur in only one source; source and label are entangled. Therefore a source-level difference cannot automatically be interpreted as a causal bias or a generalization guarantee.
- Check whether one dish or source receives systematically poorer recall, whether backgrounds/resolution appear to influence errors, and whether the model performs differently on any external Pakistani meal photos available with suitable labels and permission. If no external photos exist, state that the source-domain limitation remains untested.
- The current manifests do **not** contain person demographics. Do not invent demographic subgroup results or infer protected attributes from food images. State that demographic fairness could not be measured from this dataset; discuss applicable food, recipe, regional and source coverage instead. Note sample-size limits for every subgroup analysis.
- **Step is complete when:** the Bias Check section reports measurable subgroup evidence, explicitly marks unmeasured dimensions and explains how dataset composition may affect app users.

### Step 7: Write the results, limitations and model handoff

- Add the validation method, final selected hyperparameters, epoch history, metric definitions, tables/figures, confusion matrix, ROC-AUC analysis, error analysis and **Bias Check** to the technical report. Replace any old text that still describes the second human label review or two-model comparison as future work if it no longer reflects the actual project state. Distinguish observed results from assumptions and plans.
- Provide a short model summary for the app team: supported eight dishes, input preprocessing, output label order, checkpoint location, ranking behavior, nutrition handoff, known confusions, unsupported mixed/unfamiliar meals, and the need for user confirmation. Explain that the model estimates dish identity only; serving grams and nutrition lookup remain separate app functions.
- Test checkpoint loading and one end-to-end prediction in a clean process. Preserve the checkpoint and concise result artifacts according to the project's storage plan; ensure the app will use **the same checkpoint** that was evaluated. Update the README with repeatable training and evaluation instructions.
- **Step is complete when:** the report fulfills the Phase 4 validation, metrics, error and bias requirements, and the evaluated model is ready for app integration with documented limits.
