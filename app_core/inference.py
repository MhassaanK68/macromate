"""Validated uploads and a thread-safe adapter for the existing CLIP recognizer."""

import io
import threading
import warnings
from pathlib import Path

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "runs" / "phase4" / "best.pt"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
ALLOWED_FORMATS = ("JPEG", "PNG")


class ImageError(ValueError):
    """An upload cannot safely be processed."""


def check_image(image_bytes):
    """Decode actual content without trusting its filename or retaining the photo."""
    if not image_bytes:
        raise ImageError("No file was uploaded")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ImageError("The file is too large (maximum 10 MB)")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as image:
                if image.format not in ALLOWED_FORMATS:
                    raise ImageError("Only JPEG and PNG photos are supported")
                width, height = image.size
                if width * height > MAX_IMAGE_PIXELS:
                    raise ImageError("The image is too large (maximum 25 million pixels)")
                image.verify()
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.load()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageError("The image is too large (too many pixels)") from None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        if isinstance(error, ImageError):
            raise
        raise ImageError("This file is not a readable image") from None


_recognizer = None
_recognizer_key = None
_model_lock = threading.RLock()


def get_recognizer(checkpoint=CHECKPOINT, device="cpu"):
    """Reuse a recognizer by absolute asset identity and device; serialize loading."""
    global _recognizer, _recognizer_key
    with _model_lock:
        path = Path(checkpoint).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                "Model weights are missing. Run python scripts/prepare_app_assets.py "
                "from the project root, then restart the application."
            )
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns, str(device))
        if _recognizer is None or key != _recognizer_key:
            from models.infer import Recognizer

            _recognizer = Recognizer(str(path), device=device)
            _recognizer_key = key
        return _recognizer


def predict_dishes(image_bytes, top_k=3):
    """Return uncalibrated rankings. Each request keeps its photo/result local."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 8:
        raise ValueError("top_k must be an integer from 1 to 8")
    check_image(image_bytes)
    with _model_lock:
        recognizer = get_recognizer()
        result = recognizer.predict(io.BytesIO(image_bytes), prepared=False)
    return [
        {"dish": item["dish"], "label": item["dish"].replace(" ", "_"),
         "score": item["score"]}
        for item in result["ranked_dishes"][:top_k]
    ]
