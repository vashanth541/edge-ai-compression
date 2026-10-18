"""
Resource-conscious training pipeline for Baseline DS-CNN on Edge AI Keyword Spotting.
"""

import argparse
import csv
import os
from pathlib import Path
import random
import sys
import time
import tracemalloc
from typing import Dict, Any, Tuple

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from src.audio.config import AudioConfig, DatasetConfig
from src.audio.dataset import SpeechCommandsDataset, create_synthetic_dataset
from src.models.dscnn import DSCNN, DSCNNConfig


def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for features, targets, _ in dataloader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def evaluate_split(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on a dataloader split (validation or test)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, targets, _ in dataloader:
            features = features.to(device)
            targets = targets.to(device)

            outputs = model(features)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    eval_loss = running_loss / max(total, 1)
    eval_acc = correct / max(total, 1)
    return eval_loss, eval_acc


def run_training(config: Dict[str, Any], smoke_test: bool = False) -> Dict[str, Any]:
    """
    Execute full training workflow.

    Args:
        config: Loaded training and model configuration dictionary.
        smoke_test: If True, uses a small synthetic dataset for 2 epochs to verify execution.

    Returns:
        Summary dictionary with training results.
    """
    tracemalloc.start()
    t_start = time.time()

    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    paths_cfg = config.get("paths", {})

    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device(train_cfg.get("device", "cpu"))
    batch_size = train_cfg.get("batch_size", 16)
    lr = train_cfg.get("learning_rate", 0.001)
    weight_decay = train_cfg.get("weight_decay", 0.0001)
    epochs = 2 if smoke_test else train_cfg.get("epochs", 8)
    patience = train_cfg.get("early_stopping_patience", 3)

    data_dir = paths_cfg.get("data_dir", "data/curated")
    checkpoint_path = Path(paths_cfg.get("checkpoint_path", "models/checkpoints/baseline_dscnn.pt"))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    log_path = Path(paths_cfg.get("log_path", "results/reports/training_log.csv"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Dataset setup
    audio_cfg = AudioConfig()
    dataset_cfg = DatasetConfig()

    if smoke_test or not Path(data_dir).exists():
        temp_synthetic_dir = Path("data/smoke_synthetic")
        if not temp_synthetic_dir.exists() or len(list(temp_synthetic_dir.glob("*/*.wav"))) == 0:
            print("[INFO] Generating synthetic audio samples for smoke pipeline test...")
            create_synthetic_dataset(temp_synthetic_dir, num_speakers=15, clips_per_speaker=3)
        data_dir = str(temp_synthetic_dir)

    print(f"[INFO] Initializing dataset from: {data_dir} (Smoke test: {smoke_test})")
    train_dataset = SpeechCommandsDataset(
        data_dir=data_dir, split="train", audio_config=audio_cfg, dataset_config=dataset_cfg
    )
    val_dataset = SpeechCommandsDataset(
        data_dir=data_dir, split="val", audio_config=audio_cfg, dataset_config=dataset_cfg
    )

    print(f"[INFO] Split sizes: Train={len(train_dataset)}, Validation={len(val_dataset)}")
    if len(train_dataset) == 0:
        raise RuntimeError(f"Train dataset at {data_dir} has 0 samples.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model setup
    dscnn_config = DSCNNConfig(
        in_channels=model_cfg.get("in_channels", 1),
        in_features=model_cfg.get("in_features", 40),
        in_time_frames=model_cfg.get("in_time_frames", 47),
        num_classes=model_cfg.get("num_classes", 12),
        first_conv_channels=model_cfg.get("first_conv_channels", 28),
        ds_blocks=model_cfg.get("ds_blocks", 4),
        dropout=model_cfg.get("dropout", 0.1),
    )
    model = DSCNN(dscnn_config).to(device)
    total_params, trainable_params = model.count_parameters()
    print(f"[INFO] Baseline DS-CNN Initialized: {trainable_params:,} trainable parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize CSV log
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time_s"])

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    print(f"[INFO] Starting training for {epochs} epochs (Batch size={batch_size}, LR={lr})...")

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_split(model, val_loader, criterion, device)
        ep_dt = time.time() - ep_t0

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}% | "
            f"Time: {ep_dt:.2f}s"
        )

        # Write to log
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.5f}", f"{train_acc:.5f}", f"{val_loss:.5f}", f"{val_acc:.5f}", lr, f"{ep_dt:.2f}"])

        # Checkpoint based on validation performance
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "config": config,
                    "model_config": dscnn_config.__dict__,
                    "smoke_test": smoke_test,
                },
                checkpoint_path,
            )
            print(f"  --> Saved new best checkpoint to {checkpoint_path} (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience and not smoke_test:
                print(f"[INFO] Early stopping triggered at epoch {epoch} (no val_loss improvement for {patience} epochs).")
                break

    total_training_time = time.time() - t_start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "total_params": trainable_params,
        "total_time_s": total_training_time,
        "peak_ram_mb": peak_memory / (1024 ** 2),
        "checkpoint_path": str(checkpoint_path),
        "log_path": str(log_path),
        "smoke_test": smoke_test,
    }

    print("\n=== Training Completed ===")
    print(f"Best Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f} | Best Val Acc: {best_val_acc*100:.2f}%")
    print(f"Total Time: {total_training_time:.2f}s | Peak RAM: {summary['peak_ram_mb']:.2f} MB")
    print(f"Checkpoint: {checkpoint_path} ({checkpoint_path.stat().st_size / 1024:.2f} KB)")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train Baseline DS-CNN on Speech Commands")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to YAML training config")
    parser.add_argument("--data_dir", type=str, default=None, help="Override dataset directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--smoke_test", action="store_true", help="Run 2-epoch smoke test on synthetic data")

    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    if args.data_dir:
        config.setdefault("paths", {})["data_dir"] = args.data_dir
    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size

    run_training(config, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
