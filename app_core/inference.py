"""Photo -> top 3 dish suggestions.

This file is a thin wrapper around the model we already trained
(models/infer.py). It does not contain any new model code.

Typical use (from the app):
    suggestions = predict_dishes(uploaded_bytes)
    # [{"dish": "chicken biryani", "label": "chicken_biryani", "score": 0.93}, ...]

Run all commands from the repository root, because the model finds its cache
folder (runs/hub) relative to the project.
"""

import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# Where the trained weights are. You get this file from the weights ZIP.
CHECKPOINT = Path("runs/phase4/best.pt")

# Upload limits. These are OUR app limits, not assignment rules.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024          # 10 MB
MAX_IMAGE_PIXELS = 25_000_000                # width * height
ALLOWED_FORMATS = ("JPEG", "PNG")


class ImageError(ValueError):
    """Raised when the uploaded file is not a usable photo.

    The app can catch this and show the message to the user.
    """


def check_image(image_bytes):
    """Check an uploaded file BEFORE running the (slow) model.

    We open the real image content instead of trusting the file extension.
    Raises ImageError with a friendly message if something is wrong.
    """
    if not image_bytes:
        raise ImageError("No file was uploaded")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ImageError("The file is too large (maximum 10 MB)")

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image_format = image.format
            width, height = image.size
            image.verify()                   # detects many corrupt files
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise ImageError("This file is not a readable image")

    if image_format not in ALLOWED_FORMATS:
        raise ImageError("Only JPEG and PNG photos are supported")
    if width * height > MAX_IMAGE_PIXELS:
        raise ImageError("The image is too large (too many pixels)")


# We load the model only once and remember it here, because loading is slow.
_recognizer = None


def get_recognizer(checkpoint=CHECKPOINT, device="cpu"):
    """Load the trained model the first time, then reuse it."""
    global _recognizer
    if _recognizer is None:
        # First check the weights file, so a missing file gives a clear message.
        if not Path(checkpoint).exists():
            raise FileNotFoundError(
                f"Model weights not found at {checkpoint}. "
                "Download the weights ZIP and extract it at the repository root."
            )

        # Imported here (not at the top) so the rest of the app and the tests
        # still work on a computer where torch is not installed yet.
        from models.infer import Recognizer

        _recognizer = Recognizer(str(checkpoint), device=device)
    return _recognizer


def predict_dishes(image_bytes, top_k=3):
    """Return the top_k most likely dishes for an uploaded photo.

    Each suggestion is a dictionary:
        dish:  the model's name, e.g. "chicken biryani"
        label: the same name in CSV style, e.g. "chicken_biryani"
        score: ranking score between 0 and 1

    The scores are only a RANKING. They are not calibrated probabilities, so
    the app should call them "ranking score", not "confidence" or "accuracy".
    """
    check_image(image_bytes)
    recognizer = get_recognizer()

    # The model's loader opens the photo with PIL and does the resizing and
    # normalising itself, so we hand it the bytes as an in-memory file.
    result = recognizer.predict(io.BytesIO(image_bytes), prepared=False)

    suggestions = []
    for item in result["ranked_dishes"][:top_k]:
        suggestions.append({
            "dish": item["dish"],
            "label": item["dish"].replace(" ", "_"),
            "score": item["score"],
        })
    return suggestions
