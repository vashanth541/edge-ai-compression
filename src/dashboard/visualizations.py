"""Visualization components and plot generators for the Edge AI Compression Dashboard.

Uses Matplotlib with clean, readable aesthetics, explicit axis labels, units,
and consistent styling for embedded and mobile-targeted ML metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Consistent color palette for edge AI benchmarking
COLOR_FP32 = "#2563EB"       # Vibrant Blue
COLOR_FP32_LITE = "#60A5FA"  # Sky Blue
COLOR_INT8 = "#059669"       # Emerald Green
COLOR_ACCENT = "#D97706"     # Amber
COLOR_BG = "#F8FAFC"         # Neutral Light


def create_model_size_comparison_plot(
    fp32_checkpoint_kb: float,
    fp32_tflite_kb: Optional[float],
    int8_tflite_kb: float,
    fp32_weights_kb: float = 20.50,
    int8_weights_kb: float = 5.12,
) -> plt.Figure:
    """Create a bar chart comparing disk file sizes and raw weight payloads in KB."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=COLOR_BG)

    # 1. Total File Size on Disk
    labels1 = ["PyTorch (.pt)", "TFLite FP32", "TFLite INT8"]
    values1 = [
        fp32_checkpoint_kb,
        fp32_tflite_kb if fp32_tflite_kb is not None else 0.0,
        int8_tflite_kb,
    ]
    colors1 = [COLOR_FP32, COLOR_FP32_LITE, COLOR_INT8]

    bars1 = ax1.bar(labels1, values1, color=colors1, width=0.55, edgecolor="#1E293B", linewidth=1.2)
    ax1.set_title("On-Disk Model File Size (KB)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylabel("File Size (KB)", fontsize=10, fontweight="semibold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.set_facecolor("#FFFFFF")

    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax1.annotate(
                f"{h:.1f} KB",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )

    # 2. Raw Weights Payload (5,248 parameters)
    labels2 = ["FP32 (4 Bytes/param)", "INT8 (1 Byte/param)"]
    values2 = [fp32_weights_kb, int8_weights_kb]
    colors2 = [COLOR_FP32, COLOR_INT8]

    bars2 = ax2.bar(labels2, values2, color=colors2, width=0.45, edgecolor="#1E293B", linewidth=1.2)
    ax2.set_title("Raw Parameter Weights Payload (KB)", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylabel("Weights Size (KB)", fontsize=10, fontweight="semibold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.set_facecolor("#FFFFFF")

    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(
            f"{h:.2f} KB\n(4.00x)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    plt.tight_layout()
    return fig


def create_accuracy_f1_comparison_plot(
    fp32_acc: float,
    int8_acc: float,
    fp32_f1: float,
    int8_f1: float,
) -> plt.Figure:
    """Create a grouped bar chart comparing Test Accuracy and Macro-F1 score."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=COLOR_BG)

    metrics = ["Test Accuracy (%)", "Macro-F1 (x100)"]
    fp32_vals = [fp32_acc * 100.0, fp32_f1 * 100.0]
    int8_vals = [int8_acc * 100.0, int8_f1 * 100.0]

    x = np.arange(len(metrics))
    width = 0.32

    rects1 = ax.bar(x - width / 2, fp32_vals, width, label="FP32 Baseline (PyTorch)", color=COLOR_FP32, edgecolor="#1E293B")
    rects2 = ax.bar(x + width / 2, int8_vals, width, label="INT8 Quantized (TFLite)", color=COLOR_INT8, edgecolor="#1E293B")

    ax.set_ylabel("Score (%)", fontsize=10, fontweight="semibold")
    ax.set_title("Evaluation Performance Comparison (Held-Out Test Set: 518 Samples)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10, fontweight="semibold")
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, facecolor="#FFFFFF")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_facecolor("#FFFFFF")

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color=COLOR_INT8)

    plt.tight_layout()
    return fig


def create_latency_comparison_plot(
    fp32_latency_ms: float,
    int8_latency_ms: float,
    fp32_throughput: float,
    int8_throughput: float,
) -> plt.Figure:
    """Create side-by-side plots comparing CPU inference latency (ms) and throughput (inf/s)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0), facecolor=COLOR_BG)

    models = ["FP32 (PyTorch)", "INT8 (TFLite)"]
    latencies = [fp32_latency_ms, int8_latency_ms]
    throughputs = [fp32_throughput, int8_throughput]

    # Latency Plot
    b1 = ax1.bar(models, latencies, color=[COLOR_FP32, COLOR_INT8], width=0.45, edgecolor="#1E293B", linewidth=1.2)
    ax1.set_title("CPU Latency per Sample (ms)\n(Lower is Better)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_ylabel("Latency (milliseconds)", fontsize=10, fontweight="semibold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.set_facecolor("#FFFFFF")

    for bar in b1:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f} ms", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=9)

    # Throughput Plot
    b2 = ax2.bar(models, throughputs, color=[COLOR_FP32, COLOR_INT8], width=0.45, edgecolor="#1E293B", linewidth=1.2)
    ax2.set_title("CPU Inference Throughput (inf/sec)\n(Higher is Better)", fontsize=11, fontweight="bold", pad=10)
    ax2.set_ylabel("Throughput (inferences/sec)", fontsize=10, fontweight="semibold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.set_facecolor("#FFFFFF")

    for bar in b2:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f} inf/s", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=9)

    plt.tight_layout()
    return fig


def create_training_curves_plot(log_df: pd.DataFrame) -> Optional[plt.Figure]:
    """Generate training vs validation loss and accuracy progression curves."""
    if log_df is None or "epoch" not in log_df.columns:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=COLOR_BG)

    epochs = log_df["epoch"]

    # Loss Curve
    ax1.plot(epochs, log_df["train_loss"], marker="o", linewidth=2.0, color="#EF4444", label="Train Loss")
    ax1.plot(epochs, log_df["val_loss"], marker="s", linewidth=2.0, color="#2563EB", label="Val Loss (Checkpoint Criterion)")
    ax1.set_title("Cross-Entropy Loss Progression", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", fontsize=10, fontweight="semibold")
    ax1.set_ylabel("Loss", fontsize=10, fontweight="semibold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, facecolor="#FFFFFF")
    ax1.set_facecolor("#FFFFFF")

    # Accuracy Curve
    ax2.plot(epochs, log_df["train_acc"] * 100.0, marker="o", linewidth=2.0, color="#10B981", label="Train Accuracy (%)")
    ax2.plot(epochs, log_df["val_acc"] * 100.0, marker="s", linewidth=2.0, color="#6366F1", label="Val Accuracy (%)")
    ax2.set_title("Classification Accuracy Progression", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", fontsize=10, fontweight="semibold")
    ax2.set_ylabel("Accuracy (%)", fontsize=10, fontweight="semibold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, facecolor="#FFFFFF")
    ax2.set_facecolor("#FFFFFF")

    plt.tight_layout()
    return fig


def create_confusion_matrix_plot(
    matrix_data: List[List[int]],
    class_names: List[str],
    title: str = "INT8 Quantized DS-CNN Confusion Matrix",
) -> plt.Figure:
    """Create an annotated confusion matrix heatmap."""
    cm = np.array(matrix_data)
    fig, ax = plt.subplots(figsize=(8, 6.5), facecolor=COLOR_BG)

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="Ground Truth Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)

    # Threshold for text coloring
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j,
                i,
                format(val, "d"),
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=8,
                fontweight="bold" if i == j else "normal",
            )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def create_waveform_plot(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    title: str = "Audio Signal Waveform (1.0s Window)",
) -> plt.Figure:
    """Create an oscillogram / waveform visualization of the audio clip."""
    fig, ax = plt.subplots(figsize=(9, 2.8), facecolor=COLOR_BG)

    time_axis = np.linspace(0, len(waveform) / float(sample_rate), len(waveform))
    ax.plot(time_axis, waveform, color=COLOR_FP32, linewidth=1.0, alpha=0.9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Time (seconds)", fontsize=9, fontweight="semibold")
    ax.set_ylabel("Amplitude", fontsize=9, fontweight="semibold")
    ax.set_xlim(0, max(time_axis[len(time_axis) - 1], 1.0))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_facecolor("#FFFFFF")

    plt.tight_layout()
    return fig


def create_mfcc_plot(
    mfcc_grid: np.ndarray,
    title: str = "MFCC Feature Representation (40 coefficients × 47 frames)",
) -> plt.Figure:
    """Create a 2D heatmap of the extracted MFCC time-frequency features."""
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=COLOR_BG)

    im = ax.imshow(
        mfcc_grid,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, format="%+2.0f dB", fraction=0.03, pad=0.03)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Time Frames (20ms Hop Stride)", fontsize=9, fontweight="semibold")
    ax.set_ylabel("MFCC Coefficients (1 to 40)", fontsize=9, fontweight="semibold")

    plt.tight_layout()
    return fig


def create_probability_comparison_plot(
    class_names: List[str],
    fp32_probs: List[float],
    int8_probs: List[float],
    title: str = "Live Output Probability Distribution: FP32 vs INT8",
) -> plt.Figure:
    """Create a grouped bar chart comparing Softmax output probabilities across all 12 classes."""
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=COLOR_BG)

    x = np.arange(len(class_names))
    width = 0.36

    ax.bar(
        x - width / 2,
        [p * 100.0 for p in fp32_probs],
        width,
        label="FP32 Baseline (PyTorch)",
        color=COLOR_FP32,
        edgecolor="#1E293B",
        linewidth=0.8,
    )
    ax.bar(
        x + width / 2,
        [p * 100.0 for p in int8_probs],
        width,
        label="INT8 Quantized (TFLite)",
        color=COLOR_INT8,
        edgecolor="#1E293B",
        linewidth=0.8,
    )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.set_ylabel("Probability (%)", fontsize=9, fontweight="semibold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9, fontweight="semibold")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    plt.tight_layout()
    return fig

