# Deliverable 3 validation and release status

## Scope and evidence

The application extends repository baseline `fbbc1f469e0e` without retraining,
changing the evaluated checkpoint, modifying `models/`, or replacing saved model
evidence. Original Deliverable 1/2 reports and validation files remain historical
records. New software checks do not establish new recognition accuracy.

## Automated application checks

Verified locally on Windows, Python 3.12.14, Streamlit 1.67.0, PyTorch 2.9.0+cpu,
Transformers 4.57.1, NumPy 2.2.6, and Pillow 13.0.0.

```sh
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest tests/ -q
```

Observed result: **61 passed**; `pip check` reported **No broken requirements found**.
The test run used an explicit temporary directory outside the project to avoid
restricted-system temporary-directory permissions. This changes no application code.

A depth-one clean Git checkout of the release also passed all **61 tests** with
no `runs/` directory, local model assets, or scratch files present, and remained
clean afterward. It reused the already verified Python environment; it was not a
second dependency installation or a fresh public-host asset download.

| Area | Verified behavior |
| --- | --- |
| Nutrition | All eight unique label joins; 100 g, 150 g, fractional servings; full-precision arithmetic; source notes |
| Invalid quantities | Zero, negatives, blank/nonnumeric, NaN, infinities, booleans, and values above 5000 g are rejected |
| Uploads | JPEG/PNG accepted; grayscale/transparency accepted; empty, corrupt, truncated, unsupported, oversized, and excessive-pixel inputs rejected |
| Shared recognizer | Checkpoint/device cache identity; concurrent predictions serialized; missing-weight error |
| Asset integrity | Changed local model asset and corrupt release ZIP rejected; bounded ranges assembled exactly; failed checksum preserves existing asset |
| State | New image clears pending results; changed dish requires confirmation; changed grams invalidates estimate; cancellation blocks logging |
| Logging | Stable-ID duplicate prevention, date separation, precise totals, removal, all-date/selected-date CSV |
| Isolation | Independent workflow/log objects do not share state |
| Streamlit UI | Ready/About screens and full widget journey: recognize, correct, confirm, calculate, change grams, add, rerun, remove, cancel, export controls |

Inference unit tests and Streamlit AppTest replace model calls with explicit
test-only doubles. Production has no fake-prediction mode. Real-model integration
and browser observations are recorded separately below.

## Browser review

The local dashboard renders with the requested light/green design and readable
labels. Before upload, prediction is disabled. A corrupt `.jpg` uploaded through
the browser displayed **This file is not a readable image** and kept prediction
disabled. The initial screenshot is retained in `figures/app/dashboard.png`.

Real browser verification also completed recognition on a non-held-out biryani
photo, manual correction to chicken karahi, explicit confirmation at 150 g,
calculation, single addition, CSV download, and removal. The exported row matched
the selected dish, 150 g, source `NUT03`, 156.88 kcal, 23.94 g protein, 2.25 g
carbohydrate, and 5.69 g fat. The correction was deliberate to test that nutrition
follows the user's selected dish rather than the model's first suggestion.
After addition, the Add button and meal date were disabled. After removal, totals
returned to zero. A second browser tab opened a distinct Streamlit session with
an empty log while the first session retained its meal. This verifies separate
tab sessions locally, not cross-user isolation on the public deployment.

`figures/app/serving.png` and `figures/app/daily_log.png` show this real-model
demonstration. No fabricated predictions were used for screenshots.

## Real-model integration and timing

`python scripts/validate_app.py --output runs/app_validation.json` completed on
15 November 2026. The committed measurement record is `docs/app_validation.json`;
its per-file SHA-256 identities describe the measured application working tree
against baseline `fbbc1f469e0e`.

| Check / measurement | Observed result |
| --- | --- |
| Adapted checkpoint / pinned base cache | Both provisioned and verified |
| Direct Recognizer vs application adapter | Top-three order agrees on all ten checked photos |
| Independent CPU CLI vs adapter | Top-three order agrees on the selected sample |
| Samples | Ten non-held-out source photos; one untimed warm-up |
| CPU model initialization including imports | 14.139 s |
| Warm adapter inference, median | 0.148 s |
| Warm adapter inference, slowest of ten | 0.178 s |
| Environment | Windows 11; Python 3.12.14; Intel64 Family 6 Model 126; CPU |

These timings exclude downloads and hosted UI/network time. They are observations
on this machine, not latency guarantees or new accuracy estimates. The original
56-photo held-out evaluation was not rerun.

## Hosting status

Public deployment is awaiting account authorization. No public URL, host restart
result, or independent usability review is claimed until observed.
Streamlit's Create app flow requires the user's GitHub
sign-in/authorization; that account step is outside application source code.

Local model, browser, export, and separate-tab checks above are complete. Remaining
host acceptance checks are: deploy on Community Cloud, repeat the workflow after
a host restart, inspect resource use, verify independent users/sessions, and record
the public URL and deployed commit. Portal submission and viva participation
remain the team's responsibility.

## Repository hygiene

Only application source, configuration/dependencies, focused tests, deployment and
validation documentation, report source/PDF, and report screenshots are intended
for the release. The three overlapping initial backend scripts are consolidated
into `scripts/validate_app.py`. Local environments, model weights/cache, downloaded
archives, transfer helpers, compiler tools, test caches, temporary photos, and
LaTeX auxiliary output remain outside Git. No secrets are required or committed.

AI-assisted changes use the existing configured Git identity without changing
other contributors' history. Team members must review and understand their work;
the historical contribution checklist is not silently marked complete by automation.
