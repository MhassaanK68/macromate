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
    monkeypatch.setattr(inference, "_recognizer", FakeRecognizer())


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
