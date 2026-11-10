# Deliverable 2 submission checklist

Prepared on 11 November 2026 (Asia/Karachi). The supplied guide states a deadline
of 10 November 2026; confirm any late-submission arrangements with the instructor.

The technical deliverable is complete. Full submission readiness still depends
on the collaboration requirements below.

## Completed technical requirements

- [x] PyTorch implementation under models/, with historical model.* command compatibility.
- [x] Full model/data installation through root requirements.txt, including pinned dependencies.
- [x] Architecture, layers, trainable parameters, cross-entropy, AdamW and scheduler documented.
- [x] Six-candidate tuning table, selection rationale and fixed design choices documented.
- [x] Seeded, group-aware 280/64/56 split and leakage checks.
- [x] Accuracy, precision, recall, F1 and one-vs-rest ROC-AUC reproduced from saved predictions.
- [x] Confusion matrix, complete error analysis and explicit Bias Check.
- [x] LaTeX source and compiled Phase 3-4 PDF, with architecture and learning curves.
- [x] Contribution table based on real Git authors and changed files.
- [x] Evaluated checkpoint identity verified; large weights/base cache kept out of Git.
- [x] 22 automated tests, dependency checks and new one-epoch GPU training/checkpoint smoke test.
- [x] 424 data outputs identical across two complete runs; all 400 PNGs match original recorded hashes.
- [x] Original GPU checkpoint reproduces all 64 saved validation logit rows with maximum difference 0.
- [x] CPU CLI returns eight ranked classes; recorded single-image comparison passes within 1e-4.
- [x] Publish the verified model-weights ZIP and add its working download URL to README. Both weights and checksums were downloaded anonymously and their SHA-256 hashes matched.
- [x] Commit and push the completed sources and PDF through Git CLI (implementation commit c64639b).

## Required collaboration follow-up

- [ ] Mohid Israr (24L-2510) must contribute real code and commit it under his own identity.
  The audited history contains his source-image/CSV contribution 8210194, but no coding commit.
  Data acquisition is represented accurately in the contribution table; it has not been relabeled as coding.
- [ ] Clarify historical commit 406c823 ("Add files via upload") with the instructor because
  the guide explicitly prohibits manual GitHub uploads. Preserve history and actual authorship.
  Use Git CLI or a desktop Git client for subsequent work.

Do not manufacture another student's authorship or backdate commits. Refresh the contribution
snapshot with python scripts/collect_contributions.py and regenerate the report after genuine
team contributions are committed.

## Verified limits

- Full batched CPU replay gives the same top-1 labels on 64/64 validation images but a maximum
  logit difference of 0.324503 versus the saved GPU result. This exceeds the strict 1e-4 threshold,
  so that strict cross-device check fails. Its tolerance remains unchanged.
  Use the recorded CUDA device for exact replay; CPU inference is operational.
- The original held-out evaluation and its 54/56 result were preserved; no new held-out inference
  was run. The new one-epoch training experiment is an isolated software check, not a replacement model.
- The 400-image dataset has only seven test photos per class. Source/size confounding, absent
  demographic annotations, pending second label review and untested external meals remain disclosed.
- A graphical application, nutrition integration and daily meal logging are beyond this deliverable's
  implemented scope.

## Submission files

- Repository: https://github.com/MhassaanK68/macromate
- PDF: [Deliverable 2 report](../output/pdf/Deliverable_2_Report.pdf)
- LaTeX: [Report source](model_report.tex)
- Measurements and protocol: evidence/training.json, evidence/evaluation.json, evidence/predictions.csv.
- Test evidence: [Software validation](validation_results.json).
- Authorship evidence: [Git contribution snapshot](contributions.json).
- Local upload asset: output/weights/macromate-deliverable2-weights.zip; validated checkpoint
  SHA-256 b9089dffbfdf8c682ed28826be143a647a24a675847ca4291bf6f061c34466e4.

Published model weights: [GitHub Release](https://github.com/MhassaanK68/macromate/releases/tag/deliverable-2-weights).
