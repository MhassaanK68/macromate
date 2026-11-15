"""Provision the released checkpoint and pinned CLIP cache; verify before use."""

import hashlib
import json
import os
import tempfile
import urllib.request
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
REPOSITORY = "openai/clip-vit-base-patch32"
WEIGHTS_URL = "https://github.com/MhassaanK68/macromate/releases/download/deliverable-2-weights/macromate-deliverable2-weights.zip"
ZIP_SHA256 = "364077b4d42e2c2f7077be58b7d4fe0c4baae9d6194fa53b1e6cdd657e8b447c"
CHECKPOINT_SHA256 = "b9089dffbfdf8c682ed28826be143a647a24a675847ca4291bf6f061c34466e4"
MANIFEST = RUNS / "app_assets.json"


def download_base_file(item, snapshot, chunk_bytes=8 * 1024 * 1024):
    """Bounded, verified ranges avoid a single long-lived 600-MB transfer.

    Use four requests at a time without buffering the full model. A failed transfer never replaces the
    usable model file; a retry starts a new staged file and rechecks the full hash.
    """
    target = snapshot / item.rfilename
    expected = item.lfs.sha256
    if target.exists() and digest(target) == expected:
        return
    snapshot.mkdir(parents=True, exist_ok=True)
    size = item.size
    url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{item.rfilename}"

    def fetch(start):
        end = min(start + chunk_bytes, size) - 1
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}",
                                                               "User-Agent": "MacroMate/1.0"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise ValueError("Base-model server did not honor the requested byte range")
                    data = response.read(end - start + 2)
                if len(data) != end - start + 1:
                    raise ValueError("Incomplete base-model range")
                return start, data
            except (OSError, ValueError):
                if attempt == 2:
                    raise

    with tempfile.TemporaryDirectory(prefix="clip-transfer-", dir=RUNS) as directory:
        staged = Path(directory) / "pytorch_model.bin"
        starts = iter(range(0, size, chunk_bytes))
        with staged.open("wb") as output, ThreadPoolExecutor(max_workers=4) as pool:
            pending = {pool.submit(fetch, start) for start in [next(starts, None) for _ in range(4)]
                       if start is not None}
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    start, data = future.result()
                    output.seek(start)
                    output.write(data)
                    following = next(starts, None)
                    if following is not None:
                        pending.add(pool.submit(fetch, following))
        if staged.stat().st_size != size or digest(staged) != expected:
            raise ValueError("Pinned base model checksum mismatch; refusing to load")
        os.replace(staged, target)


def digest(path, algorithm="sha256"):
    hasher = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        if algorithm == "sha1":
            hasher.update(f"blob {Path(path).stat().st_size}\0".encode())
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def verify_assets():
    """Offline verification of all provisioned assets against the saved manifest."""
    try:
        record = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if record["revision"] != REVISION or not record["files"]:
            return False
        if digest(RUNS / "phase4" / "best.pt") != CHECKPOINT_SHA256:
            return False
        for item in record["files"]:
            path = (ROOT / item["path"]).resolve()
            if not path.is_relative_to(RUNS.resolve()) or digest(path, item["algorithm"]) != item["digest"]:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prepare_assets():
    """Idempotent setup. Do not retain download archives or put assets in Git."""
    if verify_assets():
        return
    RUNS.mkdir(parents=True, exist_ok=True)
    checkpoint = RUNS / "phase4" / "best.pt"
    if not checkpoint.exists() or digest(checkpoint) != CHECKPOINT_SHA256:
        with tempfile.TemporaryDirectory(prefix="macromate-assets-", dir=RUNS) as directory:
            archive_path = Path(directory) / "weights.zip"
            request = urllib.request.Request(WEIGHTS_URL, headers={"User-Agent": "MacroMate/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, archive_path.open("wb") as output:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
            if digest(archive_path) != ZIP_SHA256:
                raise ValueError("Released weights ZIP checksum mismatch; refusing to load")
            with zipfile.ZipFile(archive_path) as archive:
                names = [name for name in archive.namelist() if name.endswith("phase4/best.pt")]
                if len(names) != 1:
                    raise ValueError("Released weights ZIP has no unique checkpoint")
                temporary = Path(directory) / "best.pt"
                with archive.open(names[0]) as source, temporary.open("wb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                if digest(temporary) != CHECKPOINT_SHA256:
                    raise ValueError("Adapted checkpoint checksum mismatch; refusing to load")
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temporary, checkpoint)

    from huggingface_hub import HfApi, snapshot_download

    metadata = HfApi().model_info(REPOSITORY, revision=REVISION, files_metadata=True)
    selected = [item for item in metadata.siblings if item.rfilename.endswith((".json", ".txt"))
                or item.rfilename == "pytorch_model.bin"]
    snapshot = RUNS / "hub" / "models--openai--clip-vit-base-patch32" / "snapshots" / REVISION
    base = next(item for item in selected if item.rfilename == "pytorch_model.bin")
    download_base_file(base, snapshot)
    snapshot = Path(snapshot_download(REPOSITORY, revision=REVISION, cache_dir=RUNS / "hub",
                                     allow_patterns=[item.rfilename for item in selected]))
    files = []
    for item in selected:
        algorithm = "sha256" if item.lfs else "sha1"
        expected = item.lfs.sha256 if item.lfs else item.blob_id
        path = snapshot / item.rfilename
        if digest(path, algorithm) != expected:
            raise ValueError(f"Pinned base asset checksum mismatch: {item.rfilename}")
        # Preserve the snapshot path, not the resolved symlink target.
        files.append({"path": path.relative_to(ROOT).as_posix(), "algorithm": algorithm, "digest": expected})
    temporary = MANIFEST.with_suffix(".temporary")
    temporary.write_text(json.dumps({"revision": REVISION, "checkpoint_sha256": CHECKPOINT_SHA256,
                                     "files": files}, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, MANIFEST)
    if not verify_assets():
        raise ValueError("Provisioned assets failed offline verification")


if __name__ == "__main__":
    prepare_assets()
    print("Adapted checkpoint and pinned CLIP cache verified. Assets remain outside Git.")
