"""App-facing photo ranking, independent of the training module."""

import argparse
import json

import torch

from .clip_classifier import load_checkpoint
from .data import image_tensor


class Recognizer:
    def __init__(self, checkpoint, device="cpu"):
        self.device = device
        self.model, self.checkpoint = load_checkpoint(checkpoint, device)

    def predict(self, photo, prepared=False):
        pixels = image_tensor(photo, prepared=prepared).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(pixels)[0]
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite prediction scores")
        scores = logits.softmax(dim=0)
        return {"scores_are_calibrated": False, "score_description": "Softmax ranking over eight supported dishes",
                "ranked_dishes": [{"label_id": i, "dish": self.model.labels[i],
                                   "logit": float(logits[i]), "score": float(scores[i])}
                                  for i in logits.argsort(descending=True, stable=True).tolist()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("photo")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prepared", action="store_true", help="Photo is already a Phase 2 PNG")
    args = parser.parse_args()
    print(json.dumps(Recognizer(args.checkpoint, args.device).predict(args.photo, args.prepared), indent=2))


if __name__ == "__main__":
    main()
