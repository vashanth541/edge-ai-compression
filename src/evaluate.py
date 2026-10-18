"""
Evaluation and Benchmarking script for Baseline DS-CNN on held-out test data.
Computes:
- Test accuracy & Macro-F1 score
- Full 12x12 Confusion Matrix
- Per-class metrics
- Trainable parameter count & Checkpoint size (KB)
- Repeatable CPU inference latency (warm-up + 100 repetitions)
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict, Any, List, Tuple

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.audio.config import AudioConfig, DatasetConfig
from src.audio.dataset import SpeechCommandsDataset
from src.models.dscnn import DSCNN, DSCNNConfig


def benchmark_cpu_latency(
    model: nn.Module,
    sample_tensor: torch.Tensor,
    warmup_runs: int = 10,
    timed_runs: int = 100,
) -> Dict[str, float]:
    """
    Measure CPU inference latency with warm-up and repeated trials.

    Args:
        model: Evaluated PyTorch model in eval mode.
        sample_tensor: Input tensor of shape [1, 1, 40, 47].
        warmup_runs: Number of discarded initial iterations.
        timed_runs: Number of timed repetitions.

    Returns:
        Dictionary containing mean, std, min, max latency in milliseconds and throughput.
    """
    model.eval()
    times_ms: List[float] = []

    with torch.no_grad():
        # Warm-up phase
        for _ in range(warmup_runs):
            _ = model(sample_tensor)

        # Timed benchmark phase
        for _ in range(timed_runs):
            t0 = time.perf_counter()
            _ = model(sample_tensor)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    arr = np.array(times_ms)
    mean_lat = float(np.mean(arr))
    return {
        "mean_latency_ms": mean_lat,
        "std_latency_ms": float(np.std(arr)),
        "min_latency_ms": float(np.min(arr)),
        "max_latency_ms": float(np.max(arr)),
        "throughput_samples_per_sec": 1000.0 / mean_lat if mean_lat > 0 else 0.0,
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
    }


def compute_confusion_matrix_and_metrics(
    y_true: List[int], y_pred: List[int], num_classes: int = 12
) -> Tuple[np.ndarray, float, float, Dict[int, Dict[str, float]]]:
    """
    Compute 12-class confusion matrix, overall accuracy, and macro-F1 score without external deps.

    Returns:
        Tuple of (confusion_matrix, accuracy, macro_f1, per_class_metrics).
    """
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_t, y_p):
        cm[t, p] += 1

    accuracy = float(np.mean(y_t == y_p))

    f1_list = []
    per_class_metrics = {}

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        support = cm[c, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if support > 0:
            f1_list.append(f1)

        per_class_metrics[c] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "support": int(support),
        }

    macro_f1 = float(np.mean(f1_list)) if len(f1_list) > 0 else 0.0
    return cm, accuracy, macro_f1, per_class_metrics


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: Path) -> None:
    """Plot and save 12-class confusion matrix heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Baseline DS-CNN Confusion Matrix (Held-out Test Set)",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )
    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(log_csv_path: Path, output_path: Path) -> None:
    """Plot training and validation loss and accuracy curves across epochs."""
    if not log_csv_path.exists():
        return
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs, train_losses, train_accs, val_losses, val_accs = [], [], [], [], []
    with open(log_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            train_accs.append(float(row["train_acc"]) * 100.0)
            val_losses.append(float(row["val_loss"]))
            val_accs.append(float(row["val_acc"]) * 100.0)

    if not epochs:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    ax1.plot(epochs, train_losses, "o-", label="Train Loss", color="#2563eb", linewidth=2)
    ax1.plot(epochs, val_losses, "s--", label="Val Loss", color="#dc2626", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)

    ax2.plot(epochs, train_accs, "o-", label="Train Acc (%)", color="#2563eb", linewidth=2)
    ax2.plot(epochs, val_accs, "s--", label="Val Acc (%)", color="#16a34a", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)

    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_evaluation(
    checkpoint_path: str,
    data_dir: str,
    output_dir: str = "results/reports",
) -> Dict[str, Any]:
    """Execute complete evaluation on held-out test split."""
    chk_path = Path(checkpoint_path)
    if not chk_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {chk_path}")

    checkpoint = torch.load(chk_path, map_location="cpu")
    model_cfg_dict = checkpoint.get("model_config", {})
    dscnn_cfg = DSCNNConfig(**model_cfg_dict) if model_cfg_dict else DSCNNConfig()

    model = DSCNN(dscnn_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_params, trainable_params = model.count_parameters()
    chk_size_bytes = chk_path.stat().st_size
    chk_size_kb = chk_size_bytes / 1024.0

    audio_cfg = AudioConfig()
    dataset_cfg = DatasetConfig()
    idx_to_class = dataset_cfg.get_idx_to_class()

    test_dataset = SpeechCommandsDataset(
        data_dir=data_dir, split="test", audio_config=audio_cfg, dataset_config=dataset_cfg
    )

    if len(test_dataset) == 0:
        raise RuntimeError(f"No test samples found in {data_dir} for split='test'.")

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for features, targets, _ in test_loader:
            outputs = model(features)
            preds = outputs.argmax(dim=1)
            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    cm, accuracy, macro_f1, per_class = compute_confusion_matrix_and_metrics(y_true, y_pred, num_classes=12)

    # CPU Latency Benchmark on single sample
    dummy_input = torch.randn(1, 1, 40, 47)
    latency_stats = benchmark_cpu_latency(model, dummy_input, warmup_runs=10, timed_runs=100)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    metrics_data = {
        "dataset_source": data_dir,
        "is_synthetic_smoke_test": checkpoint.get("smoke_test", False),
        "test_sample_count": len(test_dataset),
        "test_accuracy": accuracy,
        "macro_f1": macro_f1,
        "trainable_parameters": trainable_params,
        "checkpoint_size_bytes": chk_size_bytes,
        "checkpoint_size_kb": chk_size_kb,
        "latency_stats": latency_stats,
        "per_class_metrics": {
            idx_to_class[c]: per_class[c] for c in per_class
        },
        "confusion_matrix": cm.tolist(),
    }

    metrics_json_file = out_path / "baseline_test_metrics.json"
    with open(metrics_json_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    # Save confusion matrix CSV
    classes_header = [idx_to_class[i] for i in range(12)]
    cm_csv_file = out_path / "baseline_confusion_matrix.csv"
    with open(cm_csv_file, "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["True\\Pred"] + classes_header)
        for i, row in enumerate(cm):
            writer.writerow([idx_to_class[i]] + row.tolist())

    # Save confusion matrix image (PNG)
    cm_png_file = out_path / "baseline_confusion_matrix.png"
    plot_confusion_matrix(cm, classes_header, cm_png_file)

    # Save training curves image (PNG) if training log exists
    log_csv_file = out_path / "training_log.csv"
    if not log_csv_file.exists():
        log_csv_file = Path("results/reports/training_log.csv")
    curves_png_file = out_path / "training_curves.png"
    plot_training_curves(log_csv_file, curves_png_file)

    # Generate Markdown Summary Report
    report_file = out_path / "baseline_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Baseline DS-CNN Evaluation Report\n\n")
        f.write(f"- **Data Source:** `{data_dir}` (Synthetic: {metrics_data['is_synthetic_smoke_test']})\n")
        f.write(f"- **Test Set Size:** {len(test_dataset)} samples\n")
        f.write(f"- **Test Accuracy:** {accuracy * 100:.2f}%\n")
        f.write(f"- **Macro-F1 Score:** {macro_f1:.4f}\n")
        f.write(f"- **Trainable Parameters:** {trainable_params:,}\n")
        f.write(f"- **Checkpoint Size:** {chk_size_kb:.2f} KB ({chk_size_bytes:,} bytes)\n")
        f.write(f"- **CPU Inference Latency (Mean):** {latency_stats['mean_latency_ms']:.2f} ms ± {latency_stats['std_latency_ms']:.2f} ms\n")
        f.write(f"- **CPU Throughput:** {latency_stats['throughput_samples_per_sec']:.1f} samples/sec\n\n")
        f.write("## Per-Class Evaluation Summary\n\n")
        f.write("| Class Name | Support | Precision | Recall | F1-Score |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for c_idx in range(12):
            c_name = idx_to_class[c_idx]
            m = per_class[c_idx]
            f.write(f"| `{c_name}` | {m['support']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1_score']:.3f} |\n")

    print("\n=== Baseline Evaluation Summary ===")
    print(f"Test Accuracy: {accuracy * 100:.2f}% | Macro-F1: {macro_f1:.4f}")
    print(f"Parameters: {trainable_params:,} | Checkpoint: {chk_size_kb:.2f} KB")
    print(f"Latency: {latency_stats['mean_latency_ms']:.2f} ms (Throughput: {latency_stats['throughput_samples_per_sec']:.1f} inf/sec)")
    print(f"Reports & Plots saved to {out_path}")
    return metrics_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate Baseline DS-CNN")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/baseline_dscnn.pt")
    parser.add_argument("--data_dir", type=str, default="data/curated")
    parser.add_argument("--output_dir", type=str, default="results/reports")

    args = parser.parse_args()
    run_evaluation(args.checkpoint, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
