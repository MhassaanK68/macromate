# Deliverable 3 validation and release status

## Simplified interface revision — 20 November 2026

The app opens on Home with the custom leaf-and-plate logo and one main Start a
meal action. Main navigation contains Home, Add a meal, and My day. Food guide
and About are under More. Photo upload starts recognition automatically; users
check dish/grams, explicitly confirm to calculate, review, and save. Each stage
shows its next action. Optional rankings, sources, and meal dates are collapsed.
Drafts survive section changes in the current session, errors offer explicit
retry, saved meals cannot be added twice, and removal offers Undo. The guide
scales recipe references by user-entered grams without logging. Shared presentation is
organized in `app_core/ui.py`; the required brand assets are `assets/logo.svg` and
`assets/styles.css`. No UI framework or dependency was added.

The revised suite passed **70 tests**; `pip check` found no broken requirements.
Checks cover staged confirmation, navigation/draft recovery, failed inference
retry without repeated automatic calls, invalid/unsupported inputs, single-save
behavior, and removal/undo. Test-only model doubles remain confined to tests.
The earlier 61-test clean-checkout and CI results below describe the original
application release. Its real-model timings and source identities remain historical
measurements; the visual redesign does not claim new model accuracy or latency.

Browser checks covered the 1440-pixel desktop and 390-pixel phone layouts.
The review screen had no horizontal overflow and metrics formed two columns on
the phone. A real non-held-out biryani photo was recognized automatically;
manual correction to karahi at 150 g produced 156.88 kcal. Explicit confirmation,
save, See my day, removal, and Undo restored the single entry and its totals.
The photo's first recognition required model initialization; this was a functional
check, not a new latency measurement or an independent usability study.
Updated report screenshots show the current interface.

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

Linux CI also passed for application commit `52cd7ae`: Ubuntu, Python 3.12,
installation from `requirements-dev.txt`, `pip check`, and the complete test suite.
The [GitHub Actions run](https://github.com/MhassaanK68/macromate/actions/runs/43722832803)
is separate from Windows timing and real-model evidence; it does not load the
released weights or establish hosted-app performance.

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

## Original application browser review (historical)

The local dashboard renders with the requested light/green design and readable
labels. Before upload, prediction is disabled. A corrupt `.jpg` uploaded through
the browser displayed **This file is not a readable image** and kept prediction
disabled. Current screenshots have replaced the original interface views.

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

The current screenshots replace the original views and show the simplified UX
review above. No fabricated predictions were used for screenshots.

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

The requested host changed from Community Cloud to **Vercel** on 20 November.
`server.py` exposes the same UI through Streamlit's ASGI API. The entry point
served Home, interactive navigation, and the configured 4 MB uploader locally
through Uvicorn. The Vercel configuration passed the current official JSON schema
checks. One regression test verifies the hosting upload limit; the model and
experiment evidence are unchanged. See `docs/deployment.md` for build packaging,
Large Functions/Fluid requirements, and bounded WebSocket session lifetime.

Public deployment is awaiting account authorization. The installed Vercel CLI
reported `login_required`. Browser sign-in reached the existing GitHub account
picker, but automatic approval review rejected continuing because it would
authorize profile/email access for Vercel. No new access was granted. This step
needs explicit user approval. No public URL, cloud build, host restart result,
or independent usability review is claimed until observed.

Local model, browser, export, and separate-tab checks above are complete. Remaining
host acceptance checks are: deploy on Vercel, repeat the workflow after
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
