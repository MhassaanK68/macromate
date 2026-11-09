"""Train and verify CLIP checkpoints using only training and validation images."""

import argparse
from collections import Counter
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
from torch.utils.data import DataLoader

from .clip_classifier import load_pretrained, save_checkpoint
from .config import (DEFAULTS, ROOT, RUNS, REVISION, CHECKPOINT, EVIDENCE, FORMAT_VERSION,
                     preprocessing, prompts_for, sha256, write_json, code_hashes, portable_path,
                     read_record, reserve_record, resolve_checkpoint, timestamp)
from .data import load_records, make_loaders, data_lineage, read_unique, DishDataset, image_tensor
from .infer import Recognizer
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


def train_run(settings, output, device="cpu", evidence=EVIDENCE):
    settings = dict(DEFAULTS, **settings)
    if not 1 <= settings["epochs"] <= 20 or settings["batch_size"] < 1 or settings["patience"] < 1:
        raise ValueError("Use 1..20 epochs and positive batch size/patience")
    if min(settings["block_lr"], settings["projection_lr"]) <= 0 or settings["weight_decay"] < 0:
        raise ValueError("Learning rates must be positive and decay nonnegative")
    if settings["dropout"] != 0:
        raise ValueError("Pretrained dropout is fixed at zero")
    output, evidence = Path(output), Path(evidence)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite existing run: {output}")
    if any((evidence / name).exists() for name in ("training.json", "evaluation.json", "predictions.csv")):
        raise ValueError("Experiment already exists; choose fresh run and evidence directories")
    records, labels = load_records()
    lineage = data_lineage(records)
    weights = RUNS / "hub" / "models--openai--clip-vit-base-patch32" / "snapshots" / REVISION / "pytorch_model.bin"
    training = {
        "format_version": FORMAT_VERSION, "status": "started", "started_at_utc": timestamp(),
        "settings": settings, "labels": labels, "prompts": prompts_for(labels),
        "preprocessing": preprocessing(), "lineage": lineage,
        "base_model": {"name": CHECKPOINT, "revision": REVISION, "sha256": sha256(weights)},
        "counts": dict(Counter(r["split"] for r in records)),
        "selection": "Highest validation top1 count, then lower loss, then earlier epoch",
        "code_sha256": code_hashes(), "checkpoint": {"path": portable_path(output / "best.pt")},
        "history": [], "verification": None,
    }
    record_path = evidence / "training.json"
    reserve_record(record_path, training)
    try:
        output.mkdir(parents=True, exist_ok=True)
        seed_everything(settings["seed"])
        train, val = make_loaders(records, settings["batch_size"], settings["seed"])
        model = load_pretrained(labels, device)
        inventory = model.trainable_inventory()
        env = environment(device)
        training.update(environment=env, trainable_inventory=inventory, trainable_count=sum(inventory.values()))
        write_json(record_path, training)
        optimizer = make_optimizer(model, settings)
        scheduler = make_scheduler(optimizer, len(train) * settings["epochs"], settings["warmup_fraction"])
        best, best_epoch, stale = None, None, 0
        start = time.perf_counter()
        for epoch in range(1, settings["epochs"] + 1):
            epoch_start = time.perf_counter()
            rates = [group["lr"] for group in optimizer.param_groups]
            train_metrics, _, _, _ = run_epoch(model, train, device, optimizer, scheduler, settings["gradient_clip"])
            val_metrics, logits, targets, ids = run_epoch(model, val, device)
            training["history"].append({"epoch": epoch, "train": train_metrics, "validation": val_metrics,
                                        "learning_rates_at_start": rates, "seconds": time.perf_counter() - epoch_start})
            print(f"{output.name} epoch {epoch}: train loss {train_metrics['loss']:.4f}; "
                  f"val {val_metrics['top1_correct']}/{val_metrics['total']}, loss {val_metrics['loss']:.4f}", flush=True)
            if best is None or selection_key(val_metrics) > selection_key(best):
                best, best_epoch, stale = val_metrics, epoch, 0
                save_checkpoint(output / "best.pt", model, {
                    "settings": settings, "lineage": lineage, "environment": env,
                    "epoch": epoch, "validation": best,
                })
                training["validation_predictions"] = {
                    "image_ids": ids, "label_ids": targets.tolist(), "logits": logits.tolist()}
            else:
                stale += 1
            write_json(record_path, training)
            if stale >= settings["patience"]:
                break
        training["result"] = {"epochs_completed": len(training["history"]), "best_epoch": best_epoch,
                              "validation": best, "seconds": time.perf_counter() - start}
        training["checkpoint"]["sha256"] = sha256(output / "best.pt")
        training["status"] = "trained"
        write_json(record_path, training)
        # Reconstruct from the saved checkpoint; keep GPU memory free for verification.
        del model, optimizer, scheduler
        verification = verify(evidence, device)
        training = read_record(record_path)
        training["result"]["reload_max_error"] = verification["reload_max_error"]
        training.update(status="completed", finished_at_utc=timestamp())
        write_json(record_path, training)
        return training
    except Exception as error:
        training = read_record(record_path)
        training.update(status="failed", failure={"at_utc": timestamp(), "error": repr(error)})
        write_json(record_path, training)
        raise


def verify(evidence=EVIDENCE, device="cpu"):
    evidence = Path(evidence)
    training = read_record(evidence / "training.json")
    if training["status"] not in ("trained", "completed"):
        raise ValueError("Training has not completed")
    torch.set_num_threads(4)
    records, labels = load_records()
    checkpoint = resolve_checkpoint(training)
    if sha256(checkpoint) != training["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint hash differs from training record")
    recognizer = Recognizer(checkpoint, device)
    payload = recognizer.checkpoint
    if (payload["labels"] != labels or labels != training["labels"]
            or payload["metadata"]["lineage"] != data_lineage(records)
            or training["lineage"] != data_lineage(records)
            or payload["metadata"]["settings"] != training["settings"]
            or payload["metadata"]["epoch"] != training["result"]["best_epoch"]):
        raise ValueError("Checkpoint no longer matches data")
    saved = training["validation_predictions"]
    loader = DataLoader(DishDataset(records, "val"), batch_size=payload["metadata"]["settings"]["batch_size"], num_workers=0)
    rows, ids, targets = [], [], []
    with torch.inference_mode():
        for pixels, truth, image_ids in loader:
            rows.append(recognizer.model(pixels.to(device)).cpu())
            ids.extend(image_ids)
            targets.extend(truth.tolist())
    logits = torch.cat(rows)
    expected = torch.tensor(saved["logits"])
    same_device = payload["metadata"]["environment"]["device"] == device
    tolerance = 1e-6 if same_device else 1e-4
    if (ids != saved["image_ids"] or targets != saved["label_ids"]
            or not torch.allclose(logits, expected, atol=tolerance, rtol=0)):
        raise ValueError("Fresh-process validation predictions do not reproduce saved best epoch")
    metadata = read_unique(ROOT / "data" / "metadata.csv", ["image_id", "local_path"])
    sample = next(r for r in records if r["image_id"] == ids[0])
    original = ROOT / metadata[ids[0]]["local_path"]
    if not torch.equal(image_tensor(original, prepared=False), image_tensor(sample["path"])):
        raise ValueError("Upload preprocessing differs from prepared input")
    prediction = recognizer.predict(original)
    if prediction["ranked_dishes"][0]["label_id"] != int(logits[0].argmax()):
        raise ValueError("App prediction disagrees with validation")
    result = {"status": "passed", "checkpoint_sha256": sha256(checkpoint), "device": device,
              "validation_images": len(ids), "reload_max_error": float((logits - expected).abs().max()),
              "absolute_tolerance": tolerance, "test_pixels_decoded": False,
              "sample_image_id": ids[0], "original_photo": str(original.relative_to(ROOT)),
              "sample_prediction": prediction}
    result["checked_at_utc"] = timestamp()
    training["verification"] = result
    write_json(evidence / "training.json", training)
    print(f"Verified {len(ids)} validation rows and original-photo app inference; max error {result['reload_max_error']}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional settings JSON; defaults are the selected pilot settings")
    parser.add_argument("--output", type=Path, default=RUNS / "phase4")
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, help="Override the configured epoch cap (1..20)")
    parser.add_argument("--verify-only", action="store_true", help="Check an existing checkpoint without training")
    args = parser.parse_args()
    if args.verify_only:
        verify(args.evidence, args.device)
        return
    import json
    settings = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    if args.epochs is not None:
        settings["epochs"] = args.epochs
    train_run(settings, args.output, args.device, args.evidence)


if __name__ == "__main__":
    main()
