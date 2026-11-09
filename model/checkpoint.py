"""Small adapted-weight checkpoints with explicit model and data contracts."""

from pathlib import Path

import torch

from .config import CHECKPOINT, REVISION, preprocessing, prompts_for
from .clip_classifier import load_pretrained


def save_checkpoint(path, model, metadata):
    required = {"settings", "lineage", "environment", "epoch", "validation"}
    if not required.issubset(metadata):
        raise ValueError("Checkpoint metadata is incomplete")
    adapted = {name: parameter.detach().cpu().clone()
               for name, parameter in model.named_parameters() if parameter.requires_grad}
    payload = {
        "format_version": 1, "checkpoint": CHECKPOINT, "revision": REVISION,
        "labels": model.labels, "prompts": model.prompts,
        "preprocessing": preprocessing(), "trainable_inventory": model.trainable_inventory(),
        "text_features": model.text_features.detach().cpu().clone(),
        "adapted_weights": adapted, "metadata": metadata,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".temporary")
    torch.save(payload, temporary)
    temporary.replace(path)


def read_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"format_version", "checkpoint", "revision", "labels", "prompts",
                "preprocessing", "trainable_inventory", "text_features", "adapted_weights", "metadata"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Checkpoint contract is incomplete")
    if (payload["format_version"] != 1 or payload["checkpoint"] != CHECKPOINT
            or payload["revision"] != REVISION or payload["preprocessing"] != preprocessing()):
        raise ValueError("Unsupported model revision or preprocessing contract")
    labels = payload["labels"]
    if len(labels) != 8 or len(set(labels)) != 8 or payload["prompts"] != prompts_for(labels):
        raise ValueError("Checkpoint label/prompt contract is invalid")
    if not {"settings", "lineage", "environment", "epoch", "validation"}.issubset(payload["metadata"]):
        raise ValueError("Checkpoint metadata is incomplete")
    return payload


def apply_checkpoint(model, payload):
    inventory = model.trainable_inventory()
    if payload["labels"] != model.labels or payload["trainable_inventory"] != inventory:
        raise ValueError("Checkpoint labels or trainable layers do not match")
    if set(payload["adapted_weights"]) != set(inventory):
        raise ValueError("Checkpoint adapted parameter names do not match")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in payload["adapted_weights"].items():
            if value.shape != parameters[name].shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid adapted tensor: {name}")
            parameters[name].copy_(value)
        features = payload["text_features"]
        if features.shape != model.text_features.shape or not torch.isfinite(features).all():
            raise ValueError("Invalid saved prompt vectors")
        model.text_features.copy_(features)
    return model


def load_checkpoint(path, device="cpu"):
    payload = read_checkpoint(path)
    model = load_pretrained(payload["labels"], device, payload["text_features"])
    apply_checkpoint(model, payload)
    model.eval()
    return model, payload
