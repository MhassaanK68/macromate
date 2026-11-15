"""Tests for app_core/inference.py that do NOT need the real model or torch.

The real model is replaced by a tiny fake, so we only test OUR code:
image checks, picking the top 3, and the label format.
Run from the repo root:  python -m pytest tests/
"""

import io

import pytest
from PIL import Image

from app_core import inference
from app_core.inference import ImageError, check_image, predict_dishes


def make_image_bytes(image_format="PNG", size=(40, 30)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "red").save(buffer, format=image_format)
    return buffer.getvalue()


class FakeRecognizer:
    """Pretends to be the model. Returns 8 dishes already ranked."""

    def predict(self, photo, prepared=False):
        assert prepared is False
        photo.read()    # the real model reads the photo too
        names = ["chicken biryani", "chicken pulao", "chicken karahi", "daal",
                 "nihari", "haleem", "samosa", "chapati"]
        return {"ranked_dishes": [{"dish": n, "score": 0.5 / (i + 1)}
                                  for i, n in enumerate(names)]}


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(inference, "get_recognizer", lambda: FakeRecognizer())


def test_png_and_jpeg_are_accepted():
    check_image(make_image_bytes("PNG"))
    check_image(make_image_bytes("JPEG"))


def test_empty_upload_is_rejected():
    with pytest.raises(ImageError):
        check_image(b"")


def test_corrupt_bytes_are_rejected():
    with pytest.raises(ImageError):
        check_image(b"this is not an image at all")


def test_truncated_image_is_rejected():
    with pytest.raises(ImageError):
        check_image(make_image_bytes("PNG")[:50])


def test_unsupported_format_is_rejected():
    with pytest.raises(ImageError):
        check_image(make_image_bytes("GIF"))


def test_oversized_file_is_rejected(monkeypatch):
    monkeypatch.setattr(inference, "MAX_UPLOAD_BYTES", 100)
    with pytest.raises(ImageError):
        check_image(make_image_bytes("PNG", size=(200, 200)))


def test_too_many_pixels_is_rejected(monkeypatch):
    monkeypatch.setattr(inference, "MAX_IMAGE_PIXELS", 100)
    with pytest.raises(ImageError):
        check_image(make_image_bytes("PNG", size=(40, 30)))


def test_predict_returns_top_three_with_both_label_styles(fake_model):
    top = predict_dishes(make_image_bytes())
    assert [s["dish"] for s in top] == ["chicken biryani", "chicken pulao", "chicken karahi"]
    assert top[0]["label"] == "chicken_biryani"
    assert top[0]["score"] > top[1]["score"] > top[2]["score"]


def test_predict_rejects_bad_upload_before_using_the_model(fake_model):
    with pytest.raises(ImageError):
        predict_dishes(b"garbage")


def test_missing_weights_give_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr(inference, "_recognizer", None)
    with pytest.raises(FileNotFoundError):
        inference.get_recognizer(checkpoint=tmp_path / "missing.pt")


def test_grayscale_and_transparent_png_are_accepted():
    for mode in ("L", "RGBA"):
        buffer = io.BytesIO()
        Image.new(mode, (20, 30)).save(buffer, "PNG")
        check_image(buffer.getvalue())


def test_recognizer_cache_tracks_checkpoint_and_device(monkeypatch, tmp_path):
    import sys
    import types

    created = []
    def construct(checkpoint, device):
        instance = object()
        created.append((checkpoint, device, instance))
        return instance

    monkeypatch.setitem(sys.modules, "models.infer", types.SimpleNamespace(Recognizer=construct))
    monkeypatch.setattr(inference, "_recognizer", None)
    monkeypatch.setattr(inference, "_recognizer_key", None)
    first, second = tmp_path / "first.pt", tmp_path / "second.pt"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    a = inference.get_recognizer(first)
    assert inference.get_recognizer(first) is a
    b = inference.get_recognizer(second)
    assert b is not a
    assert inference.get_recognizer(second, "cuda") is not b
    assert len(created) == 3


def test_shared_inference_is_serialized(monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor

    class SharedModel(FakeRecognizer):
        active = 0
        maximum = 0
        def predict(self, *args, **kwargs):
            self.active += 1
            self.maximum = max(self.active, self.maximum)
            time.sleep(.01)
            result = super().predict(*args, **kwargs)
            self.active -= 1
            return result

    shared = SharedModel()
    monkeypatch.setattr(inference, "get_recognizer", lambda: shared)
    content = make_image_bytes()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(predict_dishes, [content] * 8))
    assert shared.maximum == 1 and len(results) == 8
