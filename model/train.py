"""Plain PyTorch training; only train and validation loaders are available."""

import argparse
import math
import os
import platform
import random
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .checkpoint import save_checkpoint, read_checkpoint, apply_checkpoint
from .clip_classifier import load_pretrained
from .config import DEFAULTS, RUNS, write_json
from .data import load_records, make_loaders, data_lineage
from .metrics import summarize, selection_key


def seed_everything(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    # Small CPU batches otherwise oversubscribe laptop cores.
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def environment(device):
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: version(name) for name in
                     ("torch", "transformers", "numpy", "Pillow", "huggingface-hub", "tokenizers", "safetensors")},
        "device": str(device), "device_name": torch.cuda.get_device_name(device)
        if str(device).startswith("cuda") else platform.processor(),
        "cuda_runtime": torch.version.cuda, "precision": "float32",
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory
        if str(device).startswith("cuda") else None,
        "cpu_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def make_optimizer(model, settings):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        block = name.startswith("clip.vision_model.encoder.layers.")
        no_decay = name.endswith(".bias") or "norm" in name.lower()
        key = (block, no_decay)
        if key not in groups:
            groups[key] = {"params": [], "lr": settings["block_lr"] if block else settings["projection_lr"],
                           "weight_decay": 0.0 if no_decay else settings["weight_decay"]}
        groups[key]["params"].append(parameter)
    return torch.optim.AdamW(list(groups.values()), betas=tuple(settings["betas"]), eps=settings["epsilon"])


def make_scheduler(optimizer, steps, warmup_fraction):
    warmup = max(1, math.ceil(steps * warmup_fraction))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def run_epoch(model, loader, device, optimizer=None, scheduler=None, gradient_clip=1.0):
    training = optimizer is not None
    model.train(training)
    loss_function = nn.CrossEntropyLoss()
    all_logits, all_labels, all_ids = [], [], []
    loss_sum = 0.0
    with torch.set_grad_enabled(training):
        for pixels, labels, image_ids in loader:
            pixels, labels = pixels.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(pixels)
            loss = loss_function(logits, labels)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise ValueError("Nonfinite model logits or cross-entropy")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                         gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            loss_sum += float(loss.detach()) * len(labels)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_ids.extend(image_ids)
    logits, labels = torch.cat(all_logits), torch.cat(all_labels)
    return summarize(logits, labels, loss_sum), logits, labels, all_ids


def train_run(settings, output, device="cpu", records=None, labels=None, lineage=None):
    settings = dict(DEFAULTS, **settings)
    if not 1 <= settings["epochs"] <= 20 or settings["batch_size"] < 1 or settings["patience"] < 1:
        raise ValueError("Use 1..20 epochs and positive batch size/patience")
    if min(settings["block_lr"], settings["projection_lr"]) <= 0 or settings["weight_decay"] < 0:
        raise ValueError("Learning rates must be positive and decay nonnegative")
    if settings["dropout"] != 0:
        raise ValueError("Phase 3 fixes pretrained dropout at zero")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite existing run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(settings["seed"])
    if records is None:
        records, labels = load_records()
    if lineage is None:
        lineage = data_lineage(records)
    train, val = make_loaders(records, settings["batch_size"], settings["seed"])
    model = load_pretrained(labels, device)
    inventory = model.trainable_inventory()
    env = environment(device)
    write_json(output / "setup.json", {"settings": settings, "environment": env,
                                       "trainable_inventory": inventory, "trainable_count": sum(inventory.values())})
    optimizer = make_optimizer(model, settings)
    scheduler = make_scheduler(optimizer, len(train) * settings["epochs"], settings["warmup_fraction"])
    history, best, best_epoch, stale = [], None, None, 0
    start = time.perf_counter()
    for epoch in range(1, settings["epochs"] + 1):
        epoch_start = time.perf_counter()
        rates = [group["lr"] for group in optimizer.param_groups]
        train_metrics, _, _, _ = run_epoch(model, train, device, optimizer, scheduler, settings["gradient_clip"])
        val_metrics, logits, targets, ids = run_epoch(model, val, device)
        history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics,
                        "learning_rates_at_start": rates, "seconds": time.perf_counter() - epoch_start})
        write_json(output / "history.json", history)
        print(f"{output.name} epoch {epoch}: train loss {train_metrics['loss']:.4f}; "
              f"val {val_metrics['top1_correct']}/{val_metrics['total']}, loss {val_metrics['loss']:.4f}", flush=True)
        if best is None or selection_key(val_metrics) > selection_key(best):
            best, best_epoch, stale = val_metrics, epoch, 0
            save_checkpoint(output / "best.pt", model, {
                "settings": settings, "lineage": lineage, "environment": env,
                "epoch": epoch, "validation": best,
            })
            write_json(output / "best_validation_predictions.json", {
                "image_ids": ids, "label_ids": targets.tolist(), "logits": logits.tolist()})
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    apply_checkpoint(model, read_checkpoint(output / "best.pt"))
    reloaded_metrics, reloaded_logits, _, reloaded_ids = run_epoch(model, val, device)
    import json
    saved = json.loads((output / "best_validation_predictions.json").read_text(encoding="utf-8"))
    saved_logits = torch.tensor(saved["logits"])
    max_error = float((saved_logits - reloaded_logits).abs().max())
    if saved["image_ids"] != reloaded_ids or not torch.allclose(saved_logits, reloaded_logits, atol=1e-6, rtol=0):
        raise ValueError(f"Checkpoint reload prediction mismatch: max error {max_error}")
    result = {"settings": settings, "environment": env, "epochs_completed": len(history),
              "best_epoch": best_epoch, "validation": reloaded_metrics,
              "seconds": time.perf_counter() - start, "reload_max_error": max_error,
              "trainable_count": sum(inventory.values()), "status": "completed"}
    write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Selected settings JSON from the pilot")
    parser.add_argument("--output", type=Path, default=RUNS / "phase4")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    import json
    settings = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    settings["epochs"] = args.epochs
    train_run(settings, args.output, args.device)


if __name__ == "__main__":
    main()
