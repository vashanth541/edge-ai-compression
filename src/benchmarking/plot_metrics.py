"""
Visualization generator for training curves and confusion matrix.
Generates standalone SVG graphics without requiring matplotlib or heavy UI dependencies.
"""

import csv
import json
from pathlib import Path
import sys
from typing import List, Dict, Any

# Ensure workspace root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def generate_training_curves_svg(
    csv_log_path: str, output_svg_path: str = "results/plots/baseline_training_curves.svg"
) -> Path:
    """Generate clean SVG graph for training & validation loss and accuracy curves."""
    log_file = Path(csv_log_path)
    if not log_file.exists():
        raise FileNotFoundError(f"Training log not found: {log_file}")

    epochs, train_losses, val_losses, train_accs, val_accs = [], [], [], [], []
    with open(log_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            train_accs.append(float(row["train_acc"]) * 100.0)
            val_accs.append(float(row["val_acc"]) * 100.0)

    if not epochs:
        raise ValueError("No epoch data in log file.")

    out = Path(output_svg_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    w, h = 800, 360
    pad_l, pad_r, pad_t, pad_b = 60, 40, 50, 40
    plot_w = (w - pad_l - pad_r - 40) // 2
    plot_h = h - pad_t - pad_b

    # Scales
    max_loss = max(max(train_losses), max(val_losses)) * 1.15
    num_ep = len(epochs)

    def x_scale(ep_idx, x_offset):
        if num_ep <= 1:
            return x_offset + plot_w / 2
        return x_offset + (ep_idx / (num_ep - 1)) * plot_w

    def y_loss_scale(loss_val):
        return pad_t + plot_h - (loss_val / max(max_loss, 0.01)) * plot_h

    def y_acc_scale(acc_val):
        return pad_t + plot_h - (acc_val / 100.0) * plot_h

    x1_off = pad_l
    x2_off = pad_l + plot_w + 40

    # Build SVG content
    svg = [
        f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" style="background:#1e1e24;font-family:sans-serif;">',
        f'<text x="{w//2}" y="28" font-size="16" font-weight="bold" fill="#ffffff" text-anchor="middle">Baseline DS-CNN Training &amp; Validation Progression</text>',
    ]

    # Panel 1: Loss
    svg.append(f'<rect x="{x1_off}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="#262630" stroke="#3e3e4a"/>')
    svg.append(f'<text x="{x1_off + plot_w//2}" y="{pad_t - 10}" fill="#9e9ea8" font-size="13" text-anchor="middle">Cross-Entropy Loss</text>')
    
    # Gridlines
    for y_step in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        if y_step <= max_loss:
            y_pos = y_loss_scale(y_step)
            svg.append(f'<line x1="{x1_off}" y1="{y_pos}" x2="{x1_off+plot_w}" y2="{y_pos}" stroke="#333340" stroke-dasharray="3,3"/>')
            svg.append(f'<text x="{x1_off - 8}" y="{y_pos + 4}" fill="#7a7a85" font-size="10" text-anchor="end">{y_step:.1f}</text>')

    # Panel 2: Accuracy
    svg.append(f'<rect x="{x2_off}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="#262630" stroke="#3e3e4a"/>')
    svg.append(f'<text x="{x2_off + plot_w//2}" y="{pad_t - 10}" fill="#9e9ea8" font-size="13" text-anchor="middle">Accuracy (%)</text>')
    for acc_step in [0, 20, 40, 60, 80, 100]:
        y_pos = y_acc_scale(acc_step)
        svg.append(f'<line x1="{x2_off}" y1="{y_pos}" x2="{x2_off+plot_w}" y2="{y_pos}" stroke="#333340" stroke-dasharray="3,3"/>')
        svg.append(f'<text x="{x2_off - 8}" y="{y_pos + 4}" fill="#7a7a85" font-size="10" text-anchor="end">{acc_step}%</text>')

    # Path generators
    def pts_to_path(pts):
        return " ".join([f"{'M' if i==0 else 'L'} {x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts)])

    # Loss curves
    train_loss_pts = [(x_scale(i, x1_off), y_loss_scale(train_losses[i])) for i in range(num_ep)]
    val_loss_pts = [(x_scale(i, x1_off), y_loss_scale(val_losses[i])) for i in range(num_ep)]
    svg.append(f'<path d="{pts_to_path(train_loss_pts)}" fill="none" stroke="#4dabf7" stroke-width="2.5"/>')
    svg.append(f'<path d="{pts_to_path(val_loss_pts)}" fill="none" stroke="#ff8787" stroke-width="2.5"/>')

    # Accuracy curves
    train_acc_pts = [(x_scale(i, x2_off), y_acc_scale(train_accs[i])) for i in range(num_ep)]
    val_acc_pts = [(x_scale(i, x2_off), y_acc_scale(val_accs[i])) for i in range(num_ep)]
    svg.append(f'<path d="{pts_to_path(train_acc_pts)}" fill="none" stroke="#4dabf7" stroke-width="2.5"/>')
    svg.append(f'<path d="{pts_to_path(val_acc_pts)}" fill="none" stroke="#ff8787" stroke-width="2.5"/>')

    # Legend
    svg.append(f'<line x1="{w-220}" y1="26" x2="{w-195}" y2="26" stroke="#4dabf7" stroke-width="3"/>')
    svg.append(f'<text x="{w-190}" y="30" fill="#ced4da" font-size="11">Train</text>')
    svg.append(f'<line x1="{w-140}" y1="26" x2="{w-115}" y2="26" stroke="#ff8787" stroke-width="3"/>')
    svg.append(f'<text x="{w-110}" y="30" fill="#ced4da" font-size="11">Validation</text>')

    svg.append('</svg>')
    out.write_text("\n".join(svg), encoding="utf-8")
    print(f"[INFO] Saved training curves SVG to {out}")
    return out


def generate_confusion_matrix_svg(
    metrics_json_path: str, output_svg_path: str = "results/plots/baseline_confusion_matrix.svg"
) -> Path:
    """Generate clean SVG heatmap of 12x12 confusion matrix."""
    fpath = Path(metrics_json_path)
    if not fpath.exists():
        raise FileNotFoundError(f"Metrics JSON not found: {fpath}")

    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    cm = data.get("confusion_matrix", [])
    num_classes = len(cm)
    if num_classes != 12:
        raise ValueError(f"Expected 12x12 confusion matrix, got {num_classes}x{num_classes}")

    class_names = [
        "down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes", "_sil_", "_unk_"
    ]

    out = Path(output_svg_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cell_size = 42
    pad_l, pad_t, pad_r, pad_b = 80, 70, 30, 40
    w = pad_l + cell_size * 12 + pad_r
    h = pad_t + cell_size * 12 + pad_b

    max_val = max(max(row) for row in cm) if cm else 1

    svg = [
        f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" style="background:#1e1e24;font-family:sans-serif;">',
        f'<text x="{w//2}" y="32" font-size="16" font-weight="bold" fill="#ffffff" text-anchor="middle">Baseline DS-CNN Confusion Matrix (Held-out Test Split)</text>',
        f'<text x="{pad_l + cell_size*6}" y="54" font-size="12" fill="#adb5bd" text-anchor="middle">Predicted Class &rarr;</text>',
        f'<text x="25" y="{pad_t + cell_size*6}" font-size="12" fill="#adb5bd" text-anchor="middle" transform="rotate(-90 25 {pad_t + cell_size*6})">&larr; True Class</text>',
    ]

    # Column Headers (Predicted)
    for col_idx, name in enumerate(class_names):
        cx = pad_l + col_idx * cell_size + cell_size // 2
        svg.append(f'<text x="{cx}" y="{pad_t - 8}" font-size="10" fill="#ced4da" text-anchor="middle">{name}</text>')

    # Grid Cells
    for r_idx in range(12):
        ry = pad_t + r_idx * cell_size
        cy = ry + cell_size // 2
        # Row Header (True)
        svg.append(f'<text x="{pad_l - 8}" y="{cy + 4}" font-size="10" fill="#ced4da" text-anchor="end">{class_names[r_idx]}</text>')

        for c_idx in range(12):
            rx = pad_l + c_idx * cell_size
            val = cm[r_idx][c_idx]

            # Heatmap color: dark blue/purple to bright teal/cyan
            intensity = val / max(max_val, 1)
            if r_idx == c_idx:
                # Diagonal: turquoise / green highlight
                r = int(20 + 30 * intensity)
                g = int(50 + 170 * intensity)
                b = int(60 + 140 * intensity)
            else:
                # Off-diagonal: subtle red/purple
                r = int(35 + 160 * intensity)
                g = int(35 + 20 * intensity)
                b = int(45 + 30 * intensity)

            color = f"rgb({r},{g},{b})"
            text_color = "#ffffff" if intensity > 0.4 else "#8e8e98"

            svg.append(f'<rect x="{rx}" y="{ry}" width="{cell_size-1}" height="{cell_size-1}" fill="{color}" rx="3"/>')
            if val > 0:
                svg.append(f'<text x="{rx + cell_size//2}" y="{cy + 4}" font-size="11" font-weight="bold" fill="{text_color}" text-anchor="middle">{val}</text>')

    svg.append('</svg>')
    out.write_text("\n".join(svg), encoding="utf-8")
    print(f"[INFO] Saved confusion matrix SVG to {out}")
    return out


if __name__ == "__main__":
    generate_training_curves_svg("results/reports/training_log.csv")
    generate_confusion_matrix_svg("results/reports/baseline_test_metrics.json")
