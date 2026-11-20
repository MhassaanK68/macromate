# Application deployment

## Vercel (requested hosting target)

The `server:app` entry point exposes the same Streamlit UI through `st.App`
and ASGI. `pyproject.toml` selects that entry point, `.python-version` pins 3.12,
and `vercel.json` selects Python with Fluid Compute and a 300-second duration.
The hosting manifest pins the same dependencies as the requirements files and
selects CPU PyTorch through uv's explicit package index. Streamlit production
mode is set explicitly because Vercel vendors packages outside `site-packages`.
`buildCommand` downloads and verifies model assets before they are bundled;
runtime initialization verifies the bundle offline. Weights remain outside Git.
The upload limit is 4 MB for this entry point to fit Vercel's request-body limit.
`.vercelignore` excludes experiments, reports, tests, local caches, and dataset
images. Home's required `assets/meal.jpg` is the existing dataset photo
`RF_COMBINED_0139d55c82101638.jpg`; attribution CSVs remain included.

Import `MhassaanK68/macromate` into the intended Vercel account, using the
repository root. Enable **Fluid Compute / Active CPU** and **Large Functions**
(`VERCEL_SUPPORT_LARGE_FUNCTIONS=1`) because PyTorch and CLIP exceed the standard
Python bundle size. New projects are eligible for Large Functions by default.
Keep Python 3.12 and at least the default 2 GB function memory, then deploy.
Do not mark this deployment live until the real-photo acceptance checks pass.

The live URL is [macromate-nu.vercel.app](https://macromate-nu.vercel.app).
The production deployment was created with the authorized Vercel CLI. Automatic
deployments from GitHub pushes are not connected: the Vercel account needs a
GitHub login connection before that integration can be enabled. After signing
in with `vercel login`, use `vercel deploy --prod --yes` from the project root
for a manual release; `.vercel/` account/project metadata stays outside Git.

Vercel's WebSocket and Large Functions features are currently public beta.
Connections end at the function duration limit, and reconnection may reach a
different instance. This session-local app can therefore lose its temporary log;
CSV export is necessary for keeping meals. Test upload/media/CSV requests under
actual hosted instance routing, not only the homepage. A successful build alone
does not verify the complete app.

Official references: [Python runtime](https://vercel.com/docs/functions/runtimes/python),
[WebSockets](https://vercel.com/kb/guide/do-vercel-serverless-functions-support-websocket-connections),
and [function limits](https://vercel.com/docs/functions/limitations).

For the exact serving entry point locally:

```sh
python -m uvicorn server:app --host 127.0.0.1 --port 8502
```

## Streamlit Community Cloud

Deploy the existing GitHub repository, not a separately uploaded copy:

| Setting | Value |
| --- | --- |
| Repository | `MhassaanK68/macromate` |
| Branch | The branch containing the verified Deliverable 3 commit |
| Main file | `app.py` |
| Python | `3.12` (Advanced settings) |
| Dependency entry point | Root `requirements.txt` |
| Secrets | None needed for public model downloads |

Open [Community Cloud](https://share.streamlit.io/), select **Create app**, then
**Yup, I have an app**. Enter the settings above, confirm Python 3.12, and deploy.
The application provisions public assets on the first recognition request. This
can take several minutes after deployment or a restart; the UI shows a spinner.
Subsequent requests reuse one CPU recognizer. Model inference is serialized across
sessions, while photos, predictions, serving state, and logs remain session-local.

The host needs network access to GitHub Releases and Hugging Face, writable
ephemeral disk for `runs/`, and enough RAM for CLIP/PyTorch. Do not assume a free
host will fit: verify the actual host resource dashboard and restart behavior.
If the service exceeds its allocation, use a Python host with sufficient RAM;
never replace predictions with canned results to make deployment appear healthy.

## Local setup and recovery

Run from the repository root with Python 3.12:

```sh
python -m venv .venv
# Activate .venv using your platform's normal command.
python -m pip install -r requirements.txt
python scripts/prepare_app_assets.py
python -m streamlit run app.py
```

For automated checks, install `requirements-dev.txt` instead and run
`python -m pytest tests/ -q`. Browser UI simulation uses Streamlit AppTest and
test doubles for the recognizer; real-model verification is separate:

```sh
python scripts/validate_app.py --output runs/app_validation.json
```

This checks ten non-held-out source photos, direct Recognizer/adapter ranking
agreement, one independent CPU CLI comparison, and initialization/warm latency.
It does not retrain the model or create a new model-accuracy estimate.

## Assets and integrity

`scripts/prepare_app_assets.py` verifies the released ZIP and extracted checkpoint
against fixed SHA-256 values. It downloads only the pinned CLIP revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, checks its files against Hugging Face
Git/LFS identities, and records an ignored local manifest in `runs/app_assets.json`.
The large base binary uses four bounded range requests at a time, retries failed
ranges, and is replaced only after its complete checksum passes. This avoids the
stalled single-stream transfer encountered during local setup.
Repeat runs verify existing assets offline. The original `models/` loader remains
unchanged and still uses `local_files_only=True`.

Required paths are `runs/phase4/best.pt` and `runs/hub/`. Neither belongs in Git.
To recover from incomplete downloads, rerun the asset script. A checksum error
requires replacing the affected local download with the verified public asset;
do not disable verification. Do not delete or overwrite recorded experiment evidence.

## Release acceptance

After deployment, use a fresh browser session to upload a supported photo,
recognize it automatically, select a different supported dish, enter 150 g,
confirm to calculate, save once, and export the log. Edit the serving and verify
the previous estimate is cleared. Test a corrupt upload, select an unsupported
dish, and remove/undo a meal. Open a second
independent session and verify its empty log. Restart the service and repeat
recognition to check asset provisioning on the host. Record the actual app URL,
deployed commit, date, resource readings, and observed timings in validation notes.

The log is temporary; refresh, connection loss, hibernation, or restart can end a
session. Date selection is explicit, because the host's timezone may differ from
the user's. Uploaded photos are not written to disk, exported, or used for training.

## Report build

From the root with a TeX installation:

```sh
pdflatex -interaction=nonstopmode -halt-on-error -jobname=MacroMate_Final_Report -output-directory output/pdf docs/final_report.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=MacroMate_Final_Report -output-directory output/pdf docs/final_report.tex
```

The report uses an embedded bibliography, so no BibTeX step is needed. The portable
alternative is `tectonic -o output/pdf docs/final_report.tex`; rename its
`final_report.pdf` output to `MacroMate_Final_Report.pdf`. Keep the editable source
and that final PDF; auxiliary build products and tools remain outside Git.

Official host instructions: [deploy an app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
