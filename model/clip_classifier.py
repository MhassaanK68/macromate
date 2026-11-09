"""CLIP dish classifier and adapted-weight checkpoint saving/loading."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPModel, CLIPTokenizerFast

from .config import CHECKPOINT, REVISION, RUNS, preprocessing, prompts_for


class DishClassifier(nn.Module):
    def __init__(self, clip, labels, text_features):
        super().__init__()
        self.clip = clip
        self.labels = list(labels)
        self.prompts = prompts_for(labels)
        if text_features.shape != (8, clip.config.projection_dim):
            raise ValueError("Expected eight text features in CLIP projection space")
        if not torch.isfinite(text_features).all():
            raise ValueError("Nonfinite text features")
        self.register_buffer("text_features", F.normalize(text_features.detach().float(), dim=-1))
        self.freeze_layers()

    def freeze_layers(self):
        try:
            vision = self.clip.vision_model
            blocks = vision.encoder.layers
            final_norm = vision.post_layernorm
            projection = self.clip.visual_projection
        except AttributeError as error:
            raise ValueError("Expected CLIP vision encoder, post_layernorm and visual_projection") from error
        if len(blocks) != 12:
            raise ValueError("Expected exactly 12 CLIP vision blocks")
        for parameter in self.clip.parameters():
            parameter.requires_grad = False
        for module in (blocks[10], blocks[11], final_norm, projection):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def forward(self, pixels):
        pooled = self.clip.vision_model(pixel_values=pixels).pooler_output
        image_features = F.normalize(self.clip.visual_projection(pooled), dim=-1)
        return self.clip.logit_scale.exp() * image_features @ self.text_features.T

    def trainable_inventory(self):
        return {name: parameter.numel() for name, parameter in self.named_parameters()
                if parameter.requires_grad}


def load_pretrained(labels, device="cpu", text_features=None):
    clip = CLIPModel.from_pretrained(CHECKPOINT, revision=REVISION, cache_dir=RUNS / "hub",
                                     local_files_only=True, use_safetensors=False,
                                     attn_implementation="eager").float()
    if text_features is None:
        tokenizer = CLIPTokenizerFast.from_pretrained(CHECKPOINT, revision=REVISION,
                                                     cache_dir=RUNS / "hub", local_files_only=True)
        tokens = tokenizer(prompts_for(labels), padding=True, return_tensors="pt")
        clip.eval()
        with torch.no_grad():
            pooled = clip.text_model(**tokens).pooler_output
            text_features = clip.text_projection(pooled)
    return DishClassifier(clip, labels, text_features).to(device)


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
