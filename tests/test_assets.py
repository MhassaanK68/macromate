"""Integrity failures must block model asset use."""

import io
import json
from types import SimpleNamespace

import pytest

from scripts import prepare_app_assets as assets


def local_assets(monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(assets, "ROOT", tmp_path)
    monkeypatch.setattr(assets, "RUNS", runs)
    monkeypatch.setattr(assets, "MANIFEST", runs / "app_assets.json")
    return runs


def test_changed_base_asset_fails_offline_verification(monkeypatch, tmp_path):
    runs = local_assets(monkeypatch, tmp_path)
    checkpoint = runs / "phase4" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"adapted")
    monkeypatch.setattr(assets, "CHECKPOINT_SHA256", assets.digest(checkpoint))
    base = runs / "hub" / "pytorch_model.bin"
    base.parent.mkdir()
    base.write_bytes(b"base")
    assets.MANIFEST.write_text(json.dumps({"revision": assets.REVISION, "files": [
        {"path": "runs/hub/pytorch_model.bin", "algorithm": "sha256", "digest": assets.digest(base)}
    ]}))
    assert assets.verify_assets()
    base.write_bytes(b"changed")
    assert not assets.verify_assets()


def test_corrupt_released_zip_is_rejected_before_extracting(monkeypatch, tmp_path):
    runs = local_assets(monkeypatch, tmp_path)
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"corrupt download"))
    with pytest.raises(ValueError, match="ZIP checksum mismatch"):
        assets.prepare_assets()
    assert not (runs / "phase4" / "best.pt").exists()


def test_bounded_download_assembles_exact_ranges_and_verifies_hash(monkeypatch, tmp_path):
    import hashlib

    local_assets(monkeypatch, tmp_path)
    content = b"a small pinned base-model fixture"
    item = SimpleNamespace(rfilename="pytorch_model.bin", size=len(content),
                           lfs=SimpleNamespace(sha256=hashlib.sha256(content).hexdigest()))

    def response_for(request, timeout):
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        response = io.BytesIO(content[start:end + 1])
        response.status = 206
        response.headers = {"Content-Range": f"bytes {start}-{end}/{len(content)}"}
        return response

    monkeypatch.setattr(assets.urllib.request, "urlopen", response_for)
    snapshot = assets.RUNS / "hub" / "snapshot"
    assets.download_base_file(item, snapshot, chunk_bytes=5)
    assert (snapshot / item.rfilename).read_bytes() == content


def test_base_download_checksum_failure_preserves_existing_file(monkeypatch, tmp_path):
    import hashlib

    local_assets(monkeypatch, tmp_path)
    snapshot = assets.RUNS / "hub"
    snapshot.mkdir()
    target = snapshot / "pytorch_model.bin"
    target.write_bytes(b"old")
    item = SimpleNamespace(rfilename=target.name, size=3,
                           lfs=SimpleNamespace(sha256=hashlib.sha256(b"new").hexdigest()))
    response = io.BytesIO(b"bad")
    response.status = 206
    response.headers = {"Content-Range": "bytes 0-2/3"}
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda *a, **k: response)
    with pytest.raises(ValueError, match="checksum mismatch"):
        assets.download_base_file(item, snapshot)
    assert target.read_bytes() == b"old"
