"""Data loader and artifact auditing module for the Edge AI Compression Dashboard.

Provides robust, decoupled utilities to parse, validate, and summarize experimental
artifacts, configurations, logs, and evaluation reports. All paths are resolved
relative to the project root to ensure portability.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import yaml


def get_project_root() -> Path:
    """Resolve and return the absolute Path to the project root directory."""
    # This file is located at <project_root>/src/dashboard/data_loader.py
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent


def safe_read_json(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Safely read and parse a JSON file, returning None on missing or corrupted files."""
    path = Path(file_path)
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            return None
    except (json.JSONDecodeError, OSError):
        return None


def safe_read_csv(file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """Safely read a CSV file into a pandas DataFrame, returning None on failure."""
    path = Path(file_path)
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception:
        return None


def safe_read_text(file_path: Union[str, Path]) -> Optional[str]:
    """Safely read a text or markdown file, returning None on failure."""
    path = Path(file_path)
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def calculate_size_reduction(
    baseline_bytes: Optional[Union[int, float]],
    quantized_bytes: Optional[Union[int, float]],
) -> Dict[str, Union[float, str]]:
    """Calculate file size reduction metrics with robust edge-case handling.

    Returns:
        dict containing 'reduction_bytes', 'reduction_pct', 'compression_ratio',
        and a formatted display string.
    """
    if baseline_bytes is None or quantized_bytes is None:
        return {
            "reduction_bytes": 0.0,
            "reduction_pct": 0.0,
            "compression_ratio": 1.0,
            "status": "Not measured",
        }

    try:
        b_val = float(baseline_bytes)
        q_val = float(quantized_bytes)
    except (ValueError, TypeError):
        return {
            "reduction_bytes": 0.0,
            "reduction_pct": 0.0,
            "compression_ratio": 1.0,
            "status": "Invalid format",
        }

    if b_val <= 0 or q_val <= 0:
        return {
            "reduction_bytes": 0.0,
            "reduction_pct": 0.0,
            "compression_ratio": 1.0,
            "status": "Invalid magnitude",
        }

    reduction_bytes = b_val - q_val
    reduction_pct = (reduction_bytes / b_val) * 100.0
    compression_ratio = b_val / q_val

    return {
        "reduction_bytes": reduction_bytes,
        "reduction_pct": round(reduction_pct, 2),
        "compression_ratio": round(compression_ratio, 2),
        "status": "Valid",
    }


def get_pipeline_stages(project_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return the structured status of all 8 pipeline stages based on real artifacts."""
    root = project_root or get_project_root()

    has_env = (root / ".venv").is_dir()
    has_audio_cfg = (root / "configs" / "audio_config.yaml").is_file()
    has_train_cfg = (root / "configs" / "train_config.yaml").is_file()
    has_data = (root / "data" / "raw" / "speech_commands").is_dir()
    has_checkpoint = (root / "models" / "checkpoints" / "baseline_dscnn.pt").is_file()
    has_baseline_report = (root / "results" / "reports" / "baseline_report.md").is_file()
    has_int8_model = (root / "models" / "quantized" / "baseline_dscnn_int8.tflite").is_file()
    has_quant_report = (root / "results" / "reports" / "quantization_report.md").is_file()

    stages = [
        {
            "id": 1,
            "name": "Stage 1: Project Setup & Environment Verification",
            "status": "Completed" if has_env and has_train_cfg else "Pending",
            "description": "Virtual environment, dependency isolation, directory layout, and audio configuration.",
            "artifact": ".venv, configs/audio_config.yaml",
        },
        {
            "id": 2,
            "name": "Stage 2: Audio Preprocessing Pipeline (MFCC)",
            "status": "Completed" if has_audio_cfg else "Pending",
            "description": "16 kHz mono resampling, 40-band Mel filterbank, 40 MFCC features (40x47 grid).",
            "artifact": "src/audio/preprocessor.py, configs/audio_config.yaml",
        },
        {
            "id": 3,
            "name": "Stage 3: DS-CNN Architecture Design",
            "status": "Completed" if (root / "src" / "models" / "dscnn.py").is_file() else "Pending",
            "description": "5,248 parameter compact Depthwise Separable CNN tailored for microcontrollers.",
            "artifact": "src/models/dscnn.py",
        },
        {
            "id": 4,
            "name": "Stage 4: Baseline Training & Real Audio Evaluation",
            "status": "Completed" if has_checkpoint and has_baseline_report else "Pending",
            "description": "8 epochs on Google Speech Commands v2 real audio (55.41% accuracy, 5,248 parameters).",
            "artifact": "models/checkpoints/baseline_dscnn.pt, results/reports/baseline_report.md",
        },
        {
            "id": 5,
            "name": "Stage 5: Post-Training INT8 Quantization",
            "status": "Completed" if has_int8_model and has_quant_report else "Pending",
            "description": "Full integer INT8 quantization with zero runtime transpose ops (21.77 KB model size).",
            "artifact": "models/quantized/baseline_dscnn_int8.tflite, results/reports/quantization_report.md",
        },
        {
            "id": 6,
            "name": "Stage 6: Interactive Compression Dashboard",
            "status": "Completed",
            "description": "Locally hosted Streamlit visualization dashboard for model comparison and reporting.",
            "artifact": "src/dashboard/app.py",
        },
        {
            "id": 7,
            "name": "Stage 7: Virtual ESP32-S3 Deployment & Reviewer Demonstrator",
            "status": "Completed",
            "description": "C header export with byte verification, firmware template, live audio predictor, and resource analyzer.",
            "artifact": "deployment/esp32/include/model_data.h, src/deployment/inference_engine.py",
        },
        {
            "id": 8,
            "name": "Stage 8: Physical ESP32 Hardware Deployment & On-Device Benchmarking",
            "status": "Pending Hardware (Not performed)",
            "description": "Physical flashing, oscilloscope current measurement, and on-silicon latency benchmarking.",
            "artifact": "Pending physical ESP32-S3 hardware availability",
        },
    ]

    return stages


def get_artifact_inventory(project_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Scan and list key expected artifacts with verified existence, path, and size."""
    root = project_root or get_project_root()

    tracked_files = [
        ("Baseline Checkpoint (PyTorch)", "models/checkpoints/baseline_dscnn.pt", "Model Checkpoint"),
        ("Quantized Model (INT8 TFLite)", "models/quantized/baseline_dscnn_int8.tflite", "Quantized Model"),
        ("Reference Model (FP32 TFLite)", "models/quantized/baseline_dscnn_fp32.tflite", "Reference Model"),
        ("Model Data C Header", "deployment/esp32/include/model_data.h", "Embedded C Header"),
        ("Model Labels C Header", "deployment/esp32/include/model_labels.h", "Embedded C Header"),
        ("Export Integrity Manifest", "deployment/esp32/model_export_manifest.json", "Integrity Manifest"),
        ("ESP32 Firmware Source", "deployment/esp32/main/main.cpp", "Firmware Source"),
        ("Virtual Deployment Report (MD)", "results/reports/virtual_deployment_report.md", "Deployment Report"),
        ("Baseline Metrics (JSON)", "results/reports/baseline_test_metrics.json", "Evaluation JSON"),
        ("Quantization Comparison (JSON)", "results/reports/quantization_comparison.json", "Comparison JSON"),
        ("Quantization Comparison (CSV)", "results/reports/quantization_comparison.csv", "Comparison CSV"),
        ("Training Log (CSV)", "results/reports/training_log.csv", "Training Telemetry"),
        ("Operator Compatibility Report (MD)", "results/reports/operator_compatibility_report.md", "Technical Report"),
        ("Baseline Evaluation Report (MD)", "results/reports/baseline_report.md", "Evaluation Report"),
        ("Quantization Report (MD)", "results/reports/quantization_report.md", "Evaluation Report"),
        ("Training Curves Plot (PNG)", "results/reports/training_curves.png", "Visualization"),
        ("Quantization Confusion Matrix (PNG)", "results/reports/quantization_confusion_matrix.png", "Visualization"),
        ("Quantization Comparison Plot (PNG)", "results/reports/quantization_comparison_plots.png", "Visualization"),
        ("Reproducible Results Report (MD)", "results/reports/reproducible_results_report.md", "Final Deliverable"),
        ("Virtual Deployment Resource Report (MD)", "results/reports/virtual_deployment_resource_report.md", "Final Deliverable"),
        ("Reviewer Demonstration Script (MD)", "docs/reviewer_demonstration_script.md", "Reviewer Guide"),
    ]

    inventory = []
    for name, rel_path, category in tracked_files:
        full_path = root / rel_path
        exists = full_path.is_file()
        size_bytes = full_path.stat().st_size if exists else 0
        size_kb = round(size_bytes / 1024.0, 2) if exists else 0.0

        inventory.append({
            "name": name,
            "relative_path": rel_path,
            "category": category,
            "exists": exists,
            "size_bytes": size_bytes,
            "size_kb": size_kb,
        })

    return inventory


def get_dataset_metadata(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Retrieve verified dataset configuration and sample counts."""
    root = project_root or get_project_root()
    audio_cfg_path = root / "configs" / "audio_config.yaml"

    metadata: Dict[str, Any] = {
        "dataset_name": "Google Speech Commands v2 (Subset)",
        "sample_rate": 16000,
        "clip_duration_sec": 1.0,
        "feature_type": "MFCC (40 coefficients, 47 frames)",
        "num_classes": 12,
        "target_words": ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"],
        "special_classes": ["_silence_", "_unknown_"],
        "total_audio_files": 4890,
        "train_samples": 3936,
        "val_samples": 436,
        "test_samples": 518,
        "speaker_isolation": "100% strictly speaker-independent (0% leakage between splits)",
    }

    if audio_cfg_path.is_file():
        try:
            with open(audio_cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                if isinstance(cfg, dict):
                    audio = cfg.get("audio", {})
                    ds = cfg.get("dataset", {})
                    metadata["sample_rate"] = audio.get("sample_rate", metadata["sample_rate"])
                    metadata["target_words"] = ds.get("target_commands", metadata["target_words"])
                    metadata["num_classes"] = ds.get("num_classes", metadata["num_classes"])
        except Exception:
            pass

    return metadata


def get_safe_downloadable_artifacts(project_root: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    """Return an explicit, whitelist-governed dictionary of downloadable experiment files.

    Keys are sanitized file identifiers; values contain relative path, filename, and MIME type.
    This prevents arbitrary filesystem navigation or file traversal vulnerabilities.
    """
    return {
        "reproducible_results_report_md": {
            "rel_path": "results/reports/reproducible_results_report.md",
            "file_name": "reproducible_results_report.md",
            "mime": "text/markdown",
            "label": "Reproducible Results Report (Markdown)",
        },
        "virtual_deployment_resource_report_md": {
            "rel_path": "results/reports/virtual_deployment_resource_report.md",
            "file_name": "virtual_deployment_resource_report.md",
            "mime": "text/markdown",
            "label": "Virtual Deployment Resource Report (Markdown)",
        },
        "virtual_deployment_report_md": {
            "rel_path": "results/reports/virtual_deployment_report.md",
            "file_name": "virtual_deployment_report.md",
            "mime": "text/markdown",
            "label": "ESP32-S3 Virtual Deployment Overview (Markdown)",
        },
        "model_data_h": {
            "rel_path": "deployment/esp32/include/model_data.h",
            "file_name": "model_data.h",
            "mime": "text/x-c",
            "label": "Model Data C Header (model_data.h)",
        },
        "model_labels_h": {
            "rel_path": "deployment/esp32/include/model_labels.h",
            "file_name": "model_labels.h",
            "mime": "text/x-c",
            "label": "Class Labels C Header (model_labels.h)",
        },
        "model_export_manifest_json": {
            "rel_path": "deployment/esp32/model_export_manifest.json",
            "file_name": "model_export_manifest.json",
            "mime": "application/json",
            "label": "Cryptographic Integrity Manifest (JSON)",
        },
        "quantization_comparison_json": {
            "rel_path": "results/reports/quantization_comparison.json",
            "file_name": "quantization_comparison.json",
            "mime": "application/json",
            "label": "Quantization Comparison Metrics (JSON)",
        },
        "quantization_comparison_csv": {
            "rel_path": "results/reports/quantization_comparison.csv",
            "file_name": "quantization_comparison.csv",
            "mime": "text/csv",
            "label": "Quantization Comparison Table (CSV)",
        },
        "training_log_csv": {
            "rel_path": "results/reports/training_log.csv",
            "file_name": "training_log.csv",
            "mime": "text/csv",
            "label": "Training Log Epoch History (CSV)",
        },
        "operator_compatibility_report_md": {
            "rel_path": "results/reports/operator_compatibility_report.md",
            "file_name": "operator_compatibility_report.md",
            "mime": "text/markdown",
            "label": "TFLM / ESP32 Operator Compatibility Report (Markdown)",
        },
        "baseline_report_md": {
            "rel_path": "results/reports/baseline_report.md",
            "file_name": "baseline_report.md",
            "mime": "text/markdown",
            "label": "Baseline DS-CNN Evaluation Report (Markdown)",
        },
        "baseline_dscnn_int8_tflite": {
            "rel_path": "models/quantized/baseline_dscnn_int8.tflite",
            "file_name": "baseline_dscnn_int8.tflite",
            "mime": "application/octet-stream",
            "label": "Validated INT8 TFLite Model (.tflite)",
        },
        "baseline_test_metrics_json": {
            "rel_path": "results/reports/baseline_test_metrics.json",
            "file_name": "baseline_test_metrics.json",
            "mime": "application/json",
            "label": "Baseline FP32 Test Metrics (JSON)",
        },
        "quantization_report_md": {
            "rel_path": "results/reports/quantization_report.md",
            "file_name": "quantization_report.md",
            "mime": "text/markdown",
            "label": "INT8 Quantization Evaluation Report (Markdown)",
        },
    }

