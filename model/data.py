"""Validate Phase 2 records; pilot loaders never open reserved test photos."""

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader

from .config import SIZE, MEAN, STD
from .contracts import load_records, data_lineage


def image_tensor(path, prepared=True):
    with Image.open(path) as image:
        if prepared:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (SIZE, SIZE):
                raise ValueError("Prepared input must be an RGB 224x224 PNG")
        else:
            # Match data scripts/preprocess.py exactly, including rounding.
            image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = image.size
            scale = SIZE / min(width, height)
            width, height = max(SIZE, round(width * scale)), max(SIZE, round(height * scale))
            image = image.resize((width, height), Image.Resampling.BICUBIC)
            left, top = (width - SIZE) // 2, (height - SIZE) // 2
            image = image.crop((left, top, left + SIZE, top + SIZE))
        pixels = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    mean = np.array(MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(STD, dtype=np.float32).reshape(3, 1, 1)
    return torch.from_numpy((pixels - mean) / std)


class DishDataset(Dataset):
    def __init__(self, records, split):
        if split not in ("train", "val"):
            raise ValueError("Phase 3 datasets permit train or val only; test is reserved")
        self.records = [r for r in records if r["split"] == split]
        if not self.records:
            raise ValueError(f"No records for {split}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return image_tensor(row["path"]), row["label_id"], row["image_id"]


def make_loaders(records, batch_size, seed):
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(DishDataset(records, "train"), batch_size=batch_size,
                       shuffle=True, generator=generator, num_workers=0)
    val = DataLoader(DishDataset(records, "val"), batch_size=batch_size, num_workers=0)
    return train, val


