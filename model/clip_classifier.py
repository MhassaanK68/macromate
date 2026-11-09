"""Fixed CLIP text prototypes and a partially trainable image encoder."""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPModel, CLIPTokenizerFast

from .config import CHECKPOINT, REVISION, RUNS, prompts_for


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
