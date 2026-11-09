"""Export the Phase 3 report section directly from measured pilot evidence."""

import json

from .config import ROOT


def main():
    evidence = ROOT / "evidence" / "phase3"

    def read(name):
        return json.loads((evidence / name).read_text(encoding="utf-8"))

    results = read("results.json")
    selection = read("selection.json")
    verification = read("verification.json")
    if not selection["all_candidates_completed"] or verification["status"] != "passed":
        raise ValueError("Finish all measured pilots and fresh-process verification before exporting the report")
    chosen = next(r for r in results if r["run_id"] == selection["run_id"])
    settings, env = chosen["settings"], chosen["environment"]
    rows = []
    for result in results:
        s, v = result["settings"], result["validation"]
        name = result["run_id"].replace("_", r"\_")
        rows.append(f"{name} & {s['block_lr']:.0e} & {s['projection_lr']:.0e} & "
                    f"{s['batch_size']} & {s['weight_decay']} & {result['epochs_completed']} & "
                    f"{v['top1_correct']}/{v['total']} & {v['loss']:.4f} " + r"\\")
    text = r"""\section{Phase 3: Model Architecture and Implementation}
\subsection{Framework Selection and Design}
The engine uses PyTorch for tensors, gradients, data loaders, AdamW and checkpoints.
Hugging Face Transformers loads \path{openai/clip-vit-base-patch32} and its
matching tokenizer. Pillow reads images;
NumPy and the Python standard library handle pixels, manifests and records.
This gives direct control over partial fine-tuning without an extra training framework.
The exact revision is \texttt{3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268}.

The existing ViT-B/32 image tower receives $224\times224$ RGB inputs, uses
$32\times32$ patches, and has 12 transformer blocks of width 768 with 12 heads.
The text tower has 12 blocks of width 512 with eight heads. Both project into
512 dimensions. Eight fixed prompts, ``a photo of Pakistani [dish]'', follow
the IDs in \texttt{label\_map.json}. Normalized projected image and text vectors
produce eight cosine similarities multiplied by CLIP's frozen logit scale.
No new linear classification head is added. Scores are rankings, not calibrated confidence.

Only vision blocks 10 and 11, final vision layer normalization and visual
projection train. Embeddings, vision blocks 0--9, the text tower, text projection
and logit scale remain frozen. This limits adaptation on 280 training photos.
Prepared PNGs undergo only division by 255 and one CLIP normalization. Uploads
first receive the same EXIF correction, bicubic resize and centre crop as Phase 2.

\subsection{Loss, Optimizer and Verification}
Unweighted categorical cross-entropy uses one correct class per image, with no
label smoothing. AdamW updates trainable parameters only, with separate block
and norm/projection learning rates, betas $(0.9,0.999)$ and epsilon $10^{-8}$.
Biases and normalization weights have no weight decay. A 10\% step warmup
precedes cosine decay; gradients are clipped to global norm 1.0.
Seed 42 controls Python, NumPy and PyTorch. Validation uses no gradients or updates.
Checkpoints save adapted weights, prompt vectors, label order, preprocessing,
revision, trainable names, settings, environment, epoch and manifest/pixel hashes.
Contract checks cover joins, labels, transforms, layer updates, validation
immutability, finite outputs and checkpoint round trips. Tiny random CLIP
fixtures test mechanics only; all tuning measurements below use pretrained CLIP.

\subsection{Hyperparameter Tuning}
Six one-factor-at-a-time candidates each ran two epochs on all 280 training
and 64 validation photos, starting from identical pretrained weights and seed.
The batch-16 candidate has fewer optimizer steps per epoch; the budget holds
epochs and photo exposure equal. The 56 test photos were never decoded or scored.
Dropout stayed zero in every run. The table reports each run's best validation
epoch, selected by top-1 correct count and then lower cross-entropy.

\begin{table}[ht]
\centering
\small
\begin{tabular}{lrrrrrrr}
\toprule
Run & Block LR & Proj. LR & Batch & Decay & Epochs & Val. top-1 & Val. loss \\
\midrule
"""
    text += "\n".join(rows) + "\n"
    text += r"""\bottomrule
\end{tabular}
\caption{Measured Phase 3 pilot outcomes; validation counts are out of 64.}
\end{table}
"""
    name = chosen["run_id"].replace("_", r"\_")
    text += (f"The selected candidate is \\texttt{{{name}}}, with block LR "
             f"{settings['block_lr']:.0e}, norm/projection LR {settings['projection_lr']:.0e}, "
             f"batch {settings['batch_size']} and weight decay {settings['weight_decay']}. "
             f"It scored {chosen['validation']['top1_correct']}/64 at epoch {chosen['best_epoch']}, "
             f"with validation loss {chosen['validation']['loss']:.4f}. "
             "Highest correct count determines the run; lower loss breaks ties, followed by lower runtime. "
             "All unselected candidates follow this same declared rule. "
             "One image is 1.5625 percentage points, so small differences are exploratory.\n\n")
    by_name = {r["run_id"]: r["validation"] for r in results}
    text += (f"The baseline row is the default fine-tuning candidate, scoring "
             f"{by_name['baseline']['top1_correct']}/64. Lowering the projection rate reached "
             f"{by_name['projection_low']['top1_correct']}/64; increasing the batch to 16 reached "
             f"{by_name['batch_16']['top1_correct']}/64. Removing decay reached "
             f"{by_name['no_decay']['top1_correct']}/64 with loss {by_name['no_decay']['loss']:.6f}, "
             f"compared with default loss {by_name['baseline']['loss']:.6f}. "
             "These one-factor comparisons inform the selected learning rates, batch and decay "
             "within the short budget; near-identical decay outcomes do not establish a meaningful "
             "regularization gain. Interactions at the selected settings and statistical "
             "significance were not tested.\n\n")
    versions = ", ".join(f"{k} {env['packages'][k]}" for k in ("torch", "transformers", "numpy", "Pillow"))
    text += (f"The measured environment was Python {env['python']}; {versions}; "
             f"{env['device_name']}; CUDA runtime {env['cuda_runtime']}; float32. "
             f"There were {chosen['trainable_count']:,} trainable parameters. "
             f"Fresh-process loading reproduced all 64 validation logit rows with maximum "
             f"difference {verification['reload_max_error']:.2g}.\n\n")
    text += r"""\subsection{Phase 4 Handoff and Limits}
The pilot supports the selected learning rates, batch size and decay only
within this short budget. Zero dropout, fixed prompts, the 20-epoch full-training
cap and patience of five validation epochs are design choices, not tuned findings.
The full run starts again from pretrained weights with the selected settings;
the pilot checkpoint is not a final app model. Full training, final test metrics,
error analysis and the Bias Check remain Phase 4 work. Label review and source-domain
limitations remain unresolved. The engine recognizes dish identity only; serving
grams, nutrition lookup and the user interface remain separate later work.

The reproducible commands, run from the repository root, are:
\begin{verbatim}
python -m model.pilot --epochs 2 --device cuda
python -m model.verify --device cuda
python -m model.report
\end{verbatim}
Small evidence files are in \texttt{evidence/phase3/}; large weights remain in
ignored \texttt{runs/}. The selected full-training settings are in
\texttt{evidence/phase3/selected\_config.json}. The app-facing loader in
\texttt{model/infer.py} runs without importing the training module.
"""
    budget = read("protocol.json")["epochs_per_candidate"]
    text = text.replace("each ran two epochs", f"each ran {budget} epochs")
    text = text.replace("--epochs 2 --device cuda", f"--epochs {budget} --device cuda")
    (ROOT / "report" / "phase3.tex").write_text(text, encoding="utf-8")
    lines = ["# Phase 3 measured results", "", "Generated from `evidence/phase3/` by `python -m model.report`.", ""]
    for result in results:
        v = result["validation"]
        lines.append(f"- `{result['run_id']}`: {v['top1_correct']}/{v['total']} validation top-1, "
                     f"loss {v['loss']:.6f}; best epoch {result['best_epoch']}; "
                     f"{result['epochs_completed']} completed epochs; {result['seconds']:.1f} seconds.")
    lines.extend(["", f"Selected: `{chosen['run_id']}`. " + selection["reason"], "", selection["limits"], "",
                  f"Fresh-process reload: maximum logit error {verification['reload_max_error']}; "
                  "all 64 validation photos checked. No reserved test photos decoded or scored.", "",
                  "The selected configuration is `evidence/phase3/selected_config.json`. "
                  "The pilot checkpoint is `" + selection["pilot_checkpoint"] + "`. "
                  "Use the README Phase 4 command to begin full training from the pretrained base."])
    (ROOT / "docs" / "05_Phase_3_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote report/phase3.tex and docs/05_Phase_3_results.md")


if __name__ == "__main__":
    main()
