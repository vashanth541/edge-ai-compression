"""Edge AI Model Compression Pipeline - Interactive Streamlit Dashboard.

Locally hosted dashboard providing comprehensive visualization, comparison,
live audio inference lab, and virtual ESP32-S3 deployment demonstration
for embedded keyword spotting.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.data_loader import (
    calculate_size_reduction,
    get_artifact_inventory,
    get_dataset_metadata,
    get_pipeline_stages,
    get_project_root,
    get_safe_downloadable_artifacts,
    safe_read_csv,
    safe_read_json,
    safe_read_text,
)
from src.dashboard.visualizations import (
    create_accuracy_f1_comparison_plot,
    create_confusion_matrix_plot,
    create_latency_comparison_plot,
    create_mfcc_plot,
    create_model_size_comparison_plot,
    create_probability_comparison_plot,
    create_training_curves_plot,
    create_waveform_plot,
)
from src.deployment.inference_engine import AudioValidationError, ModelPredictor
from src.deployment.resource_analyzer import ResourceAnalyzer, analyze_model_resources
from src.dashboard.model_analyzer import (
    TFLiteAnalysisError,
    analyze_tflite_bytes,
    analyze_tflite_file,
    format_analysis_for_display,
)
from src.dashboard.compatibility_checker import (
    check_esp32s3_compatibility,
    generate_compatibility_markdown,
    ESP32S3_PROFILE,
)
from src.dashboard.package_builder import (
    build_deployment_package,
    get_verified_deployment_package,
)


# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="Edge AI Model Compression Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-verified {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-pending {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-skipped {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
    }
    table.dataframe {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
        margin-top: 8px;
        margin-bottom: 16px;
    }
    table.dataframe th {
        background-color: #F1F5F9;
        color: #0F172A;
        padding: 10px 12px;
        border-bottom: 2px solid #CBD5E1;
        text-align: left;
        font-weight: 600;
    }
    table.dataframe td {
        padding: 8px 12px;
        border-bottom: 1px solid #E2E8F0;
        color: #334155;
    }
    table.dataframe tr:hover {
        background-color: #F8FAFC;
    }
    .inference-box {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 16px;
    }
    .prediction-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_table(df: pd.DataFrame) -> None:
    """Render a table safely via clean HTML without requiring PyArrow or NumPy 2.0."""
    html_code = df.to_html(index=False, classes="dataframe", border=0)
    st.markdown(f'<div style="overflow-x: auto;">{html_code}</div>', unsafe_allow_html=True)


@st.cache_resource
def get_cached_model_predictor(root_path: Path) -> ModelPredictor:
    """Cache model predictor instance in Streamlit session."""
    return ModelPredictor(project_root=root_path)


@st.cache_data(show_spinner=False)
def get_cached_deployment_package(project_root_str: str) -> Dict[str, Any]:
    """
    Cached package provider and validator for Streamlit Community Cloud and local use.
    Loads or generates the ESP32-S3 INT8 deployment package and verifies all cryptographic
    checksums and manifest entries prior to serving the download button.
    """
    return get_verified_deployment_package(project_root_str)


def render_prominent_download_banner(root_path: Path) -> None:
    """Renders the top hero card with the 1-click ESP32-S3 INT8 deployment package download."""
    package_info = get_cached_deployment_package(str(root_path))

    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
                    color: white; padding: 22px 24px; border-radius: 12px; margin-bottom: 20px;
                    border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
                <div>
                    <h2 style="color: #F8FAFC; margin: 0 0 6px 0; font-size: 1.45rem;">
                        ⚡ Download ESP32-S3 INT8 Deployment Package
                    </h2>
                    <p style="color: #94A3B8; margin: 0; font-size: 0.95rem;">
                        Verified full-integer INT8 DS-CNN model, 64-bit aligned C headers, and ESP32-S3 TFLM firmware scaffold.
                    </p>
                </div>
                <div style="background: #0EA5E9; color: white; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85rem;">
                    Target: ESP32-S3 (Xtensa LX7 @ 240 MHz)
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if package_info["valid"]:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Model Binary Size", f"{package_info['model_size_kb']:.2f} KB", f"{package_info['model_size_bytes']:,} B")
        with m2:
            st.metric("Compression Ratio", "1.31×", "-23.60% vs FP32")
        with m3:
            st.metric("Test Accuracy", "54.83%", "-0.58 pp vs FP32")
        with m4:
            st.metric("Deployment Bundle", f"{package_info['zip_size_kb']:.1f} KB", f"{package_info['zip_size_bytes']:,} B ZIP")

        st.download_button(
            label="⬇️ Download ESP32-S3 INT8 Deployment Package (.zip)",
            data=package_info["zip_bytes"],
            file_name=package_info["filename"],
            mime="application/zip",
            type="primary",
            use_container_width=True,
            key="prominent_hero_download_btn",
            help="One-click download of the complete validated ESP32-S3 deployment package (.zip).",
        )

        with st.expander("📋 Package Contents & Verification Manifest (8 Artifacts)", expanded=False):
            st.markdown(
                """
                - **`models/baseline_dscnn_int8.tflite`** — Fully quantized INT8 TFLite model (`22,296` bytes)
                - **`include/model_data.h`** — C byte-array header with `alignas(8)` memory alignment
                - **`include/model_labels.h`** — Class label string array (12 speech command classes)
                - **`main.cpp`** — Complete TFLM inference scaffold template (`AllOpsResolver`, `30 KB` arena)
                - **`compatibility_report.md`** — Static hardware analysis (Flash: 0.5%, SRAM: 6.5%)
                - **`deployment_report.json`** — Machine-readable metrics, metadata, and status disclosures
                - **`README.md`** — Step-by-step ESP-IDF build, compile, and flashing guide
                - **`MANIFEST.json`** — Cryptographic SHA-256 checksums and exact byte sizes for all files
                """
            )
            st.caption(f"Verified Model SHA-256: `{package_info['model_sha256']}`")
            st.caption("✅ Integrity check: 100% of internal package files match cryptographic manifest checksums.")
    else:
        st.error(f"❌ Deployment package unavailable: {package_info['error']}")

    st.markdown("---")


# ==============================================================================
# Data Loading & Session State
# ==============================================================================
project_root = get_project_root()

# Artifact file paths
quant_json_path = project_root / "results" / "reports" / "quantization_comparison.json"
baseline_json_path = project_root / "results" / "reports" / "baseline_test_metrics.json"
train_log_path = project_root / "results" / "reports" / "training_log.csv"
operator_report_path = project_root / "results" / "reports" / "operator_compatibility_report.md"
virtual_deploy_report_path = project_root / "results" / "reports" / "virtual_deployment_report.md"

quant_data = safe_read_json(quant_json_path)
baseline_data = safe_read_json(baseline_json_path)
train_log_df = safe_read_csv(train_log_path)
operator_md = safe_read_text(operator_report_path)
virtual_deploy_md = safe_read_text(virtual_deploy_report_path)
dataset_meta = get_dataset_metadata(project_root)
pipeline_stages = get_pipeline_stages(project_root)


# ==============================================================================
# Sidebar Navigation & Controls
# ==============================================================================
with st.sidebar:
    st.title("⚡ Edge AI Compression")
    st.caption("Microcontroller Keyword Spotting Pipeline")
    st.markdown("---")

    st.subheader("Navigation")
    section = st.radio(
        "Jump to Section:",
        [
            "🔧 Compression Framework",
            "Overview & Status",
            "Audio Inference Lab",
            "Model Comparison",
            "Visualizations",
            "Virtual ESP32 Deployment",
            "Artifacts & Reports",
        ],
        index=0,
    )

    st.markdown("---")
    st.subheader("Reviewer Environment")
    st.text("Host: Windows 11 (CPU-only)")
    st.text("CPU: AMD Ryzen 3 7320U")
    st.text("Python: 3.11 (64-bit)")
    st.text("Runtimes: PyTorch + TFLite")
    st.text("Target: ESP32-S3 (Virtual)")

    st.markdown("---")
    if st.button("🔄 Refresh Data & Cache", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()


# ==============================================================================
# Prominent Hero Section: Download ESP32-S3 INT8 Deployment Package
# ==============================================================================
render_prominent_download_banner(project_root)


# ==============================================================================
# SECTION A: Project Overview & Pipeline Stages
# ==============================================================================
if section == "Overview & Status":
    st.header("📌 Project Overview & Pipeline Status")
    st.markdown(
        """
        This dashboard presents the experimental progress of the **Edge AI Model Compression Pipeline**.
        The objective is to train, quantize, and validate an ultra-compact **Depthwise Separable CNN (DS-CNN)**
        for 12-class Keyword Spotting (KWS) to deploy onto resource-constrained microcontrollers (ESP32-S3).
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Model Architecture", value="DS-CNN (5,248 params)")
    with col2:
        st.metric(label="Dataset Sample Count", value=f"{dataset_meta['total_audio_files']:,} WAVs")
    with col3:
        st.metric(label="Audio Feature", value="MFCC (40×47)")
    with col4:
        st.metric(label="Classes", value=f"{dataset_meta['num_classes']} Classes")

    st.markdown("### 🛠️ Pipeline Stages & Execution Status")
    
    for stage in pipeline_stages:
        status_badge = (
            "✅ **Completed**" if stage["status"] == "Completed" else "⏳ **Pending Review**"
        )
        with st.expander(f"{stage['name']} — {status_badge}", expanded=(stage["id"] in [4, 5, 7])):
            st.markdown(f"**Description:** {stage['description']}")
            st.markdown(f"**Key Artifacts:** `{stage['artifact']}`")

    st.markdown("### 🎙️ Verified Dataset Metadata & Split Policy")
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.markdown(
            f"""
            - **Dataset Source:** `{dataset_meta['dataset_name']}`
            - **Audio Sample Rate:** {dataset_meta['sample_rate']} Hz (Mono)
            - **Audio Window:** {dataset_meta['clip_duration_sec']} second (16,000 samples)
            - **Input Feature Representation:** {dataset_meta['feature_type']}
            - **Target Commands (10):** `{', '.join(dataset_meta['target_words'])}`
            - **Special Classes (2):** `{', '.join(dataset_meta['special_classes'])}`
            """
        )
    with d_col2:
        st.markdown(
            f"""
            - **Total Verified Audio Clips:** {dataset_meta['total_audio_files']:,}
            - **Training Split:** {dataset_meta['train_samples']:,} samples (829 unique speakers)
            - **Validation Split:** {dataset_meta['val_samples']:,} samples (98 unique speakers)
            - **Held-Out Test Split:** {dataset_meta['test_samples']:,} samples (102 unique speakers)
            - **Speaker Isolation:** {dataset_meta['speaker_isolation']}
            """
        )


# ==============================================================================
# SECTION B: Live Audio Inference Lab
# ==============================================================================
elif section == "Audio Inference Lab":
    st.header("🎙️ Live Audio Inference Lab (Reviewer Testing)")
    st.markdown(
        """
        Test the compressed keyword spotting models interactively on genuine audio clips or upload your own `.wav` file.
        The lab executes real pre-processing (MFCC feature extraction) and runs live inference on **both**
        the FP32 baseline and INT8 quantized models.
        """
    )

    st.info(
        "💻 **Inference Execution Notice:**\n"
        "Inferences executed in this lab run live on your **Laptop CPU (AMD Ryzen 3 7320U)** using PyTorch CPU "
        "and TensorFlow Lite Runtime. **No physical microcontroller is connected.** "
        "Single-sample inference times shown below reflect laptop CPU performance."
    )

    sample_audio_dir = project_root / "data" / "sample_audio"
    sample_files = sorted(list(sample_audio_dir.glob("*.wav"))) if sample_audio_dir.is_dir() else []

    audio_source_type = st.radio(
        "Select Audio Source:",
        ["Bundled Speech Commands Sample (12 classes)", "Upload Custom WAV File"],
        horizontal=True,
    )

    selected_audio_bytes: Optional[bytes] = None
    selected_filename: str = ""

    if audio_source_type == "Bundled Speech Commands Sample (12 classes)":
        if sample_files:
            file_options = {f.name: f for f in sample_files}
            selected_name = st.selectbox(
                "Choose a sample keyword audio file:",
                list(file_options.keys()),
                index=list(file_options.keys()).index("yes.wav") if "yes.wav" in file_options else 0,
            )
            selected_path = file_options[selected_name]
            selected_filename = selected_name
            with open(selected_path, "rb") as f:
                selected_audio_bytes = f.read()
        else:
            st.warning("No sample audio files found in `data/sample_audio/`.")
    else:
        uploaded_file = st.file_uploader(
            "Upload a 1-second WAV audio file (16 kHz recommended, mono or stereo, max 10 MB):",
            type=["wav"],
        )
        if uploaded_file is not None:
            selected_audio_bytes = uploaded_file.getvalue()
            selected_filename = uploaded_file.name

    if selected_audio_bytes is not None:
        st.markdown("---")
        st.subheader(f"Audio Inspection: `{selected_filename}`")

        predictor = get_cached_model_predictor(project_root)

        # 1. Validation & Properties
        try:
            raw_waveform, sr = predictor.validate_audio_input(selected_audio_bytes)
            duration_s = len(raw_waveform) / float(sr)

            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            with p_col1:
                st.metric("Sample Rate", f"{sr:,} Hz")
            with p_col2:
                st.metric("Duration", f"{duration_s:.2f} s")
            with p_col3:
                st.metric("Channels", "Mono (converted)")
            with p_col4:
                st.metric("Payload Size", f"{len(selected_audio_bytes) / 1024.0:.1f} KB")

            st.audio(selected_audio_bytes, format="audio/wav")

            # 2. Live Model Inference & Feature Extraction
            with st.spinner("Executing real-time preprocessing & inference on Laptop CPU..."):
                pred_res = predictor.predict(selected_audio_bytes)

            mfcc_grid = pred_res["mfcc_grid"]
            pt_info = pred_res["fp32_pytorch"]
            int8_info = pred_res["int8_tflite"]
            agreed = pred_res["models_agreed"]

            # Waveform & MFCC Visualizations
            st.markdown("### 📊 Audio Waveform & Extracted MFCC Spectrogram")
            v_col1, v_col2 = st.columns(2)

            with v_col1:
                w_fig = create_waveform_plot(raw_waveform, sr, title=f"Waveform: {selected_filename}")
                st.pyplot(w_fig)

            with v_col2:
                m_fig = create_mfcc_plot(mfcc_grid, title="Extracted MFCC Feature (40 Mel × 47 Frames)")
                st.pyplot(m_fig)

            # 3. Live Model Inference Comparison
            st.markdown("### ⚡ Live Model Inference Comparison")

            if agreed:
                st.success(
                    f"✅ **Prediction Agreement:** Both FP32 and INT8 models predicted: "
                    f"**`{int8_info['predicted_label']}`**"
                )
            else:
                st.warning(
                    f"⚠️ **Prediction Disagreement:** "
                    f"FP32 predicted **`{pt_info['predicted_label']}`** ({pt_info['confidence']*100:.1f}%), "
                    f"whereas INT8 predicted **`{int8_info['predicted_label']}`** ({int8_info['confidence']*100:.1f}%)."
                )

            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.markdown("#### 🔹 PyTorch Baseline (FP32)")
                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <h3 style="margin-top:0; color:#1E3A8A;">Label: <code>{pt_info['predicted_label']}</code></h3>
                        <p><strong>Confidence:</strong> {pt_info['confidence'] * 100:.2f}%</p>
                        <p><strong>Laptop CPU Latency:</strong> {pt_info['latency_ms']:.2f} ms</p>
                        <p><strong>Top-3 Classes:</strong></p>
                        <ul>
                            <li><strong>{pt_info['top3'][0]['label']}</strong>: {pt_info['top3'][0]['prob']*100:.1f}%</li>
                            <li>{pt_info['top3'][1]['label']}: {pt_info['top3'][1]['prob']*100:.1f}%</li>
                            <li>{pt_info['top3'][2]['label']}: {pt_info['top3'][2]['prob']*100:.1f}%</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with res_col2:
                st.markdown("#### 🔸 TensorFlow Lite (Full INT8)")
                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <h3 style="margin-top:0; color:#065F46;">Label: <code>{int8_info['predicted_label']}</code></h3>
                        <p><strong>Confidence:</strong> {int8_info['confidence'] * 100:.2f}%</p>
                        <p><strong>Laptop CPU Latency:</strong> {int8_info['latency_ms']:.2f} ms</p>
                        <p><strong>Top-3 Classes:</strong></p>
                        <ul>
                            <li><strong>{int8_info['top3'][0]['label']}</strong>: {int8_info['top3'][0]['prob']*100:.1f}%</li>
                            <li>{int8_info['top3'][1]['label']}: {int8_info['top3'][1]['prob']*100:.1f}%</li>
                            <li>{int8_info['top3'][2]['label']}: {int8_info['top3'][2]['prob']*100:.1f}%</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Probability Distribution Comparison Chart
                st.markdown("### 📈 Class Probability Distribution Comparison")
                prob_fig = create_probability_comparison_plot(
                    fp32_probs=np.array(pt_info["all_probabilities"]),
                    int8_probs=np.array(int8_info["all_probabilities"]),
                    class_names=pred_res["class_names"],
                )
                st.pyplot(prob_fig)

                st.caption(
                    "Note: Single-sample testing above demonstrates live pipeline execution. "
                    "For full test-set evaluation (518 samples), see the **Model Comparison** section."
                )

        except AudioValidationError as e:
            st.error(f"❌ **Audio Validation Error:** {e}")
        except Exception as e:
            st.error(f"❌ **Inference Error:** {e}")


# ==============================================================================
# SECTION C: Model Comparison (FP32 Baseline vs INT8 Quantized)
# ==============================================================================
elif section == "Model Comparison":
    st.header("⚖️ Model Comparison: FP32 Baseline vs Full INT8 Quantized")

    if quant_data is None:
        st.warning("⚠️ Quantization comparison metrics (`quantization_comparison.json`) not found.")
    else:
        models_dict = quant_data.get("models", {})
        comp_summary = quant_data.get("comparison", {})

        fp32_info = models_dict.get("fp32_pytorch", {})
        int8_info = models_dict.get("int8_tflite", {})

        # Top metric cards
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        fp32_acc = fp32_info.get("test_accuracy", 0.0)
        int8_acc = int8_info.get("test_accuracy", 0.0)
        acc_delta = comp_summary.get("accuracy_change_pp", 0.0)

        fp32_size_kb = fp32_info.get("file_size_kb", 0.0)
        int8_size_kb = int8_info.get("file_size_kb", 0.0)
        size_red_pct = comp_summary.get("file_size_reduction_pct", 0.0)

        fp32_lat = fp32_info.get("latency_mean_ms", 0.0)
        int8_lat = int8_info.get("latency_mean_ms", 0.0)
        speedup = comp_summary.get("speedup_ratio", 1.0)

        with m_col1:
            st.metric(
                label="Test Accuracy (518 Samples)",
                value=f"{int8_acc * 100:.2f}% (INT8)",
                delta=f"{acc_delta:+.2f} pp vs FP32 ({fp32_acc * 100:.2f}%)",
            )
        with m_col2:
            st.metric(
                label="Macro-F1 Score",
                value=f"{int8_info.get('macro_f1', 0.0):.4f}",
                delta=f"{comp_summary.get('macro_f1_change', 0.0):+.4f} vs FP32 ({fp32_info.get('macro_f1', 0.0):.4f})",
            )
        with m_col3:
            st.metric(
                label="On-Disk Model File Size",
                value=f"{int8_size_kb:.2f} KB",
                delta=f"-{size_red_pct:.1f}% reduction (4.73x)",
                delta_color="inverse",
            )
        with m_col4:
            st.metric(
                label="CPU Latency (AMD Ryzen 3)",
                value=f"{int8_lat:.2f} ms",
                delta=f"{speedup:.2f}x speedup vs {fp32_lat:.2f} ms",
                delta_color="inverse",
            )

        st.markdown("### 📊 Comprehensive Side-by-Side Comparison Table")
        comparison_rows = [
            {
                "Attribute / Metric": "Format / Runtime",
                "FP32 Baseline": "PyTorch State Dict (.pt)",
                "INT8 Quantized": "TensorFlow Lite FlatBuffer (.tflite)",
                "Comparison / Delta": "TFLM Embedded Target",
                "Measurement Type": "Architecture",
            },
            {
                "Attribute / Metric": "Test Accuracy (518 samples)",
                "FP32 Baseline": f"{fp32_acc * 100:.2f}% (287 / 518)",
                "INT8 Quantized": f"{int8_acc * 100:.2f}% (284 / 518)",
                "Comparison / Delta": f"{acc_delta:+.2f} percentage points",
                "Measurement Type": "Measured",
            },
            {
                "Attribute / Metric": "Macro-F1 Score",
                "FP32 Baseline": f"{fp32_info.get('macro_f1', 0.0):.4f}",
                "INT8 Quantized": f"{int8_info.get('macro_f1', 0.0):.4f}",
                "Comparison / Delta": f"{comp_summary.get('macro_f1_change', 0.0):+.4f}",
                "Measurement Type": "Measured",
            },
            {
                "Attribute / Metric": "Total Trainable Parameters",
                "FP32 Baseline": "5,248 parameters (FP32)",
                "INT8 Quantized": "5,248 parameters (INT8)",
                "Comparison / Delta": "Identical topology",
                "Measurement Type": "Architecture",
            },
            {
                "Attribute / Metric": "Raw Weights Payload",
                "FP32 Baseline": "20.50 KB (5,248 × 4 Bytes)",
                "INT8 Quantized": "5.12 KB (5,248 × 1 Byte)",
                "Comparison / Delta": "4.00x exact reduction",
                "Measurement Type": "Measured",
            },
            {
                "Attribute / Metric": "Total File Size on Disk",
                "FP32 Baseline": f"{fp32_size_kb:.2f} KB (105,396 Bytes)",
                "INT8 Quantized": f"{int8_size_kb:.2f} KB (22,296 Bytes)",
                "Comparison / Delta": f"{size_red_pct:.1f}% reduction (4.73x)",
                "Measurement Type": "Measured",
            },
            {
                "Attribute / Metric": "CPU Inference Latency (Mean)",
                "FP32 Baseline": f"{fp32_lat:.2f} ms ± {fp32_info.get('latency_std_ms', 0.0):.2f} ms",
                "INT8 Quantized": f"{int8_lat:.2f} ms ± {int8_info.get('latency_std_ms', 0.0):.2f} ms",
                "Comparison / Delta": f"{speedup:.2f}x speedup",
                "Measurement Type": "Measured",
            },
            {
                "Attribute / Metric": "CPU Throughput",
                "FP32 Baseline": f"{fp32_info.get('throughput_samples_per_sec', 0.0):.1f} samples/sec",
                "INT8 Quantized": f"{int8_info.get('throughput_samples_per_sec', 0.0):.1f} samples/sec",
                "Comparison / Delta": f"+{int8_info.get('throughput_samples_per_sec', 0.0) - fp32_info.get('throughput_samples_per_sec', 0.0):.1f} samples/sec",
                "Measurement Type": "Measured",
            },
            {
                "Attribute / Metric": "Input Tensor Representation",
                "FP32 Baseline": "float32 [1, 1, 40, 47]",
                "INT8 Quantized": "int8 [1, 40, 47, 1] (scale: 0.4796, zp: 54)",
                "Comparison / Delta": "Pure integer input",
                "Measurement Type": "Verified",
            },
            {
                "Attribute / Metric": "Output Tensor Representation",
                "FP32 Baseline": "float32 [1, 12]",
                "INT8 Quantized": "int8 [1, 12] (scale: 0.1009, zp: 82)",
                "Comparison / Delta": "Pure integer output",
                "Measurement Type": "Verified",
            },
        ]
        render_table(pd.DataFrame(comparison_rows))

        st.info(
            "ℹ️ **Measurement Methodology Note:**\n"
            "- Laptop latency was benchmarked on a Windows 11 host (AMD Ryzen 3 7320U CPU, single-threaded execution).\n"
            "- Protocol: 10 warm-up runs followed by 100 timed repetitions on identical test spectrograms.\n"
            "- Both models were evaluated on the exact same 518 held-out test audio samples from Google Speech Commands v2."
        )

        st.markdown("### 🎯 Per-Class Precision, Recall, and F1 Comparison")
        fp32_classes = fp32_info.get("per_class_metrics", {})
        int8_classes = int8_info.get("per_class_metrics", {})

        class_rows = []
        for cname in sorted(dataset_meta["target_words"] + dataset_meta["special_classes"]):
            fp32_c = fp32_classes.get(cname, {})
            int8_c = int8_classes.get(cname, {})
            supp = fp32_c.get("support", int8_c.get("support", 0))

            f1_fp32 = fp32_c.get("f1_score", 0.0)
            f1_int8 = int8_c.get("f1_score", 0.0)

            class_rows.append({
                "Class": cname,
                "Support": supp,
                "FP32 Precision": f"{fp32_c.get('precision', 0.0):.3f}",
                "FP32 Recall": f"{fp32_c.get('recall', 0.0):.3f}",
                "FP32 F1": f"{f1_fp32:.3f}",
                "INT8 Precision": f"{int8_c.get('precision', 0.0):.3f}",
                "INT8 Recall": f"{int8_c.get('recall', 0.0):.3f}",
                "INT8 F1": f"{f1_int8:.3f}",
                "F1 Impact": f"{f1_int8 - f1_fp32:+.3f}",
            })
        render_table(pd.DataFrame(class_rows))


# ==============================================================================
# SECTION D: Visualizations
# ==============================================================================
elif section == "Visualizations":
    st.header("📈 Experiment Visualizations & Analytics")

    v_tab1, v_tab2, v_tab3, v_tab4 = st.tabs([
        "Compression & Accuracy",
        "Latency & Throughput",
        "Training Curves",
        "Confusion Matrix",
    ])

    with v_tab1:
        st.subheader("Model Size and Accuracy Benchmarks")
        if quant_data:
            fp32_info = quant_data["models"]["fp32_pytorch"]
            int8_info = quant_data["models"]["int8_tflite"]

            c_fig1 = create_model_size_comparison_plot(
                fp32_checkpoint_kb=fp32_info.get("file_size_kb", 102.93),
                fp32_tflite_kb=28.50,
                int8_tflite_kb=int8_info.get("file_size_kb", 21.77),
                fp32_weights_kb=20.50,
                int8_weights_kb=5.12,
            )
            st.pyplot(c_fig1)

            c_fig2 = create_accuracy_f1_comparison_plot(
                fp32_acc=fp32_info.get("test_accuracy", 0.5541),
                int8_acc=int8_info.get("test_accuracy", 0.5483),
                fp32_f1=fp32_info.get("macro_f1", 0.5110),
                int8_f1=int8_info.get("macro_f1", 0.5179),
            )
            st.pyplot(c_fig2)
        else:
            st.info("Quantization data not available for plotting.")

    with v_tab2:
        st.subheader("Inference Latency & Throughput (Laptop CPU)")
        if quant_data:
            fp32_info = quant_data["models"]["fp32_pytorch"]
            int8_info = quant_data["models"]["int8_tflite"]

            l_fig = create_latency_comparison_plot(
                fp32_latency_ms=fp32_info.get("latency_mean_ms", 2.29),
                int8_latency_ms=int8_info.get("latency_mean_ms", 0.30),
                fp32_throughput=fp32_info.get("throughput_samples_per_sec", 436.5),
                int8_throughput=int8_info.get("throughput_samples_per_sec", 3312.3),
            )
            st.pyplot(l_fig)
        else:
            st.info("Latency data not available for plotting.")

    with v_tab3:
        st.subheader("Baseline DS-CNN Training & Validation Curves (8 Epochs)")
        if train_log_df is not None:
            t_fig = create_training_curves_plot(train_log_df)
            if t_fig:
                st.pyplot(t_fig)
            render_table(train_log_df)
        else:
            st.info("Training log (`training_log.csv`) not found.")

    with v_tab4:
        st.subheader("Classification Confusion Matrix")
        if quant_data:
            cm_model = st.radio(
                "Select Model for Confusion Matrix:",
                ["Full INT8 Quantized (TFLite)", "FP32 Baseline (PyTorch)"],
                horizontal=True,
            )
            selected_key = "int8_tflite" if "INT8" in cm_model else "fp32_pytorch"
            cm_data = quant_data["models"][selected_key].get("confusion_matrix")
            all_classes = sorted(dataset_meta["target_words"] + dataset_meta["special_classes"])

            if cm_data:
                cm_fig = create_confusion_matrix_plot(
                    matrix_data=cm_data,
                    class_names=all_classes,
                    title=f"{cm_model} Confusion Matrix (518 Test Samples)",
                )
                st.pyplot(cm_fig)
            else:
                st.info("Confusion matrix data missing from report.")


# ==============================================================================
# SECTION E: Virtual ESP32-S3 Embedded Deployment
# ==============================================================================
elif section == "Virtual ESP32 Deployment":
    st.header("⚡ Virtual ESP32-S3 Embedded Deployment Demonstrator")
    st.warning(
        "⚠️ **CRITICAL REVIEWER NOTICE: SOFTWARE-ONLY VIRTUAL DEPLOYMENT**\n\n"
        "All firmware project files, C array exports, tensor arena calculations, and resource estimates "
        "are **software-only analytical projections** for an Espressif ESP32-S3 microcontroller. "
        "**Physical hardware flashing, on-chip cycle timing, and power measurements have NOT been performed.**"
    )

    res_data = analyze_model_resources(project_root)

    # 1. Visual 7-Stage Workflow
    st.subheader("🛠️ 7-Stage Virtual Deployment Pipeline Status")

    wf_steps = [
        {
            "Step": "1. INT8 Model Quantization",
            "Status": "✅ Verified",
            "Details": "22,296 bytes FlatBuffer validated; 54.83% accuracy (-0.58 pp delta vs FP32).",
        },
        {
            "Step": "2. C-Array Header Generation",
            "Status": "✅ Verified (Exact)",
            "Details": "Generated `model_data.h` (alignas(8)); SHA-256 byte-for-byte integrity verified.",
        },
        {
            "Step": "3. Operator Inspection",
            "Status": "✅ Verified",
            "Details": "5 required kernels (CONV_2D, DEPTHWISE_CONV_2D, AVERAGE_POOL_2D, FULLY_CONNECTED, RESHAPE) compatible with TFLM & esp-nn.",
        },
        {
            "Step": "4. Virtual Resource Sizing",
            "Status": "✅ Computed",
            "Details": "Model Flash: 21.77 KB (99.45% free); Arena RAM: 30.0 KB; Total SRAM: 52.0 KB (89.8% headroom).",
        },
        {
            "Step": "5. Firmware Template Prepared",
            "Status": "✅ Prepared",
            "Details": "Complete ESP-IDF project under `deployment/esp32/` (CMakeLists, sdkconfig, idf_component.yml, main.cpp).",
        },
        {
            "Step": "6. Firmware Toolchain Build",
            "Status": "⚠️ Template Only",
            "Details": "Espressif ESP-IDF (`idf.py`) is not installed on laptop host. Firmware template verified syntactically.",
        },
        {
            "Step": "7. Physical Hardware Deployment",
            "Status": "🛑 Not Performed",
            "Details": "Physical ESP32-S3 board flashing, current draw, and on-chip latency explicitly marked Not Performed.",
        },
    ]

    for step in wf_steps:
        badge = step["Status"]
        expanded = "⚠️" in badge or "🛑" in badge or "Exact" in badge
        with st.expander(f"{step['Step']} — {badge}", expanded=expanded):
            st.markdown(f"**Verification Outcome:** {step['Details']}")

    st.markdown("---")
    st.subheader("📊 ESP32-S3 Resource Allocation & Headroom")

    ms = res_data["model_storage"]
    fp = res_data["flash_partition"]
    ta = res_data["tensor_arena"]
    sb = res_data["sram_ram_budget"]

    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    with r_col1:
        st.metric(label="Flash Occupancy (.rodata)", value=f"{ms['compiled_rodata_kb']} KB", help="8-byte aligned C array payload")
        st.caption(f"{fp['total_flash_occupancy_pct']}% of 4 MB SPI Flash")
    with r_col2:
        st.metric(label="Tensor Arena RAM", value=f"{ta['configured_arena_kb']} KB", help="Double-buffered intermediate activations + margin")
        st.caption(f"{ta['safety_margin_pct']}% engineering safety margin")
    with r_col3:
        st.metric(label="Total Estimated SRAM", value=f"{sb['total_estimated_sram_kb']} KB", help="Arena + FreeRTOS stack + I2S DMA + MFCC scratchpad")
        st.caption(f"{sb['sram_occupancy_pct']}% of 512 KB internal SRAM")
    with r_col4:
        st.metric(label="Available SRAM Headroom", value=f"{sb['sram_headroom_kb']} KB", help="Free memory for Wi-Fi/BLE and user tasks")
        st.caption(f"{sb['sram_headroom_pct']}% headroom free")

    st.markdown("### 📋 Detailed Subsystem RAM Allocations")
    sram_rows = [
        {
            "Subsystem / Allocation": "Tensor Arena (Intermediate Activations)",
            "Allocation (Bytes)": f"{sb['tensor_arena_bytes']:,} B",
            "Allocation (KB)": f"{sb['tensor_arena_kb']} KB",
            "Internal SRAM Occupancy": "5.86%",
            "Memory Type": "Internal Static RAM",
        },
        {
            "Subsystem / Allocation": "FreeRTOS Application / Model Task Stack",
            "Allocation (Bytes)": f"{sb['main_task_stack_bytes']:,} B",
            "Allocation (KB)": f"{sb['main_task_stack_kb']} KB",
            "Internal SRAM Occupancy": "3.13%",
            "Memory Type": "Internal Static RAM",
        },
        {
            "Subsystem / Allocation": "I2S Audio DMA Buffers (Ping-Pong)",
            "Allocation (Bytes)": f"{sb['i2s_dma_buffers_bytes']:,} B",
            "Allocation (KB)": f"{sb['i2s_dma_buffers_kb']} KB",
            "Internal SRAM Occupancy": "0.39%",
            "Memory Type": "DMA-capable SRAM",
        },
        {
            "Subsystem / Allocation": "MFCC Audio Preprocessor Scratchpad",
            "Allocation (Bytes)": f"{sb['mfcc_dsp_scratch_bytes']:,} B",
            "Allocation (KB)": f"{sb['mfcc_dsp_scratch_kb']} KB",
            "Internal SRAM Occupancy": "0.78%",
            "Memory Type": "Internal Static RAM",
        },
        {
            "Subsystem / Allocation": "Total Estimated Working RAM",
            "Allocation (Bytes)": f"{sb['total_estimated_sram_bytes']:,} B",
            "Allocation (KB)": f"{sb['total_estimated_sram_kb']} KB",
            "Internal SRAM Occupancy": f"{sb['sram_occupancy_pct']}%",
            "Memory Type": "52.0 KB out of 512 KB",
        },
        {
            "Subsystem / Allocation": "Available SRAM Headroom",
            "Allocation (Bytes)": f"{512 * 1024 - sb['total_estimated_sram_bytes']:,} B",
            "Allocation (KB)": f"{sb['sram_headroom_kb']} KB",
            "Internal SRAM Occupancy": f"{sb['sram_headroom_pct']}% Free",
            "Memory Type": "Available for Network & App",
        },
    ]
    render_table(pd.DataFrame(sram_rows))

    st.markdown("### 🧱 Layer-by-Layer Activation Tensor Sizing")
    layer_rows = []
    layers_list = res_data["tensor_arena"]["layers_breakdown"]
    peak_bytes = max(l["bytes"] for l in layers_list)
    for idx, l in enumerate(layers_list, 1):
        layer_rows.append({
            "Layer Index": idx,
            "Layer Name / Operator": l["layer"],
            "Output Dimensions": str(l["shape"]),
            "Data Type": l["dtype"],
            "Buffer Size (Bytes)": f"{l['bytes']:,} B",
            "Buffer Size (KB)": f"{round(l['bytes'] / 1024.0, 2)} KB",
            "Peak Layer": "⭐ Yes" if l["bytes"] == peak_bytes else "No",
        })
    render_table(pd.DataFrame(layer_rows))

    st.markdown("### 🔍 Technical Assumptions & Safety Margins")
    st.markdown(
        """
        1. **Tensor Arena Memory Reuse:** TFLM uses a greedy memory planner where intermediate buffers are reused
           across non-overlapping execution phases. The minimum required arena is the sum of the two largest adjacent
           activation buffers (`13,440 B + 13,440 B = 26,880 B`) plus FlatBuffer parsing overhead (~2 KB), yielding `28.9 KB`.
           We configured a `30.0 KB` arena (+10.7% margin) to ensure zero memory exhaustion.
        2. **Flash Occupancy:** The compiled `.rodata` array occupies `22,296 bytes` (21.77 KB). In a typical 4 MB Flash configuration,
           this consumes less than `0.55%` of total Flash, leaving abundant space for user applications and OTA updates.
        3. **Host Toolchain Limitation:** The ESP-IDF toolchain (`idf.py`) is not present on this Windows machine.
           The firmware template under `deployment/esp32/` is complete and verified syntactically, but compilation is marked as
           **Template Only / Not Built** until the Espressif toolchain is installed.
        4. **Hardware Latency:** Literature indicates ~6–10 ms per 1-second audio window on ESP32-S3 @ 240 MHz using `esp-nn` assembly vector extensions.
           Physical on-chip latency and power consumption have **not been measured**.
        """
    )


# ==============================================================================
# SECTION F: Experiment Artifacts & Reports Explorer
# ==============================================================================
elif section == "Artifacts & Reports":
    st.header("📁 Experiment Artifacts & Reports Explorer")
    st.markdown("Inspect, verify, and download genuine experiment artifacts produced by the pipeline.")

    st.subheader("Tracked Project Artifacts Inventory")
    inventory = get_artifact_inventory(project_root)
    inv_df = pd.DataFrame(inventory)
    render_table(inv_df[["name", "category", "relative_path", "size_kb", "exists"]])

    st.markdown("---")
    st.subheader("📥 Download Verified Artifacts & Deployment Files")
    safe_files = get_safe_downloadable_artifacts(project_root)

    dl_cols = st.columns(3)
    idx = 0
    for key, f_info in safe_files.items():
        f_path = project_root / f_info["rel_path"]
        with dl_cols[idx % 3]:
            if f_path.is_file():
                with open(f_path, "rb") as fh:
                    data = fh.read()
                st.download_button(
                    label=f"⬇️ {f_info['label']}",
                    data=data,
                    file_name=f_info["file_name"],
                    mime=f_info["mime"],
                    use_container_width=True,
                )
            else:
                st.button(f"❌ {f_info['file_name']} (Missing)", disabled=True, use_container_width=True)
        idx += 1

    st.markdown("---")
    st.subheader("📄 Interactive Report Viewer")
    report_to_view = st.selectbox(
        "Select Report to View:",
        [
            "Reproducible Results Report (Markdown)",
            "Virtual ESP32 Resource Report (Markdown)",
            "Virtual ESP32 Deployment Report (Markdown)",
            "Operator Compatibility Report (Markdown)",
            "Quantization Comparison Report (Markdown)",
            "Baseline Evaluation Report (Markdown)",
        ],
    )

    if "Reproducible Results" in report_to_view:
        content = safe_read_text(project_root / "results" / "reports" / "reproducible_results_report.md")
        if content:
            st.markdown(content)
        else:
            st.info("Reproducible results report not found.")
    elif "Virtual ESP32 Resource" in report_to_view:
        content = safe_read_text(project_root / "results" / "reports" / "virtual_deployment_resource_report.md")
        if content:
            st.markdown(content)
        else:
            st.info("Virtual deployment resource report not found.")
    elif "Virtual ESP32 Deployment" in report_to_view:
        content = safe_read_text(virtual_deploy_report_path)
        if content:
            st.markdown(content)
        else:
            st.info("Virtual deployment report not found.")
    elif "Operator Compatibility" in report_to_view:
        content = safe_read_text(operator_report_path)
        if content:
            st.markdown(content)
        else:
            st.info("Operator compatibility report not found.")
    elif "Quantization Comparison" in report_to_view:
        content = safe_read_text(project_root / "results" / "reports" / "quantization_report.md")
        if content:
            st.markdown(content)
        else:
            st.info("Quantization report not found.")
    elif "Baseline Evaluation" in report_to_view:
        content = safe_read_text(project_root / "results" / "reports" / "baseline_report.md")
        if content:
            st.markdown(content)
        else:
            st.info("Baseline report not found.")


# ==============================================================================
# SECTION: Compression Framework
# ==============================================================================
if section == "🔧 Compression Framework":
    st.header("🔧 Automated AI Model Compression & ESP32-S3 Deployment Framework")
    st.markdown(
        """
        Select the built-in DS-CNN demonstration model or upload a `.tflite` model file.
        Inspect the model, check ESP32-S3 compatibility, run the available optimization
        workflow, and download a complete deployment package.
        """
    )
    st.info(
        "**Scope note:** This framework performs real TFLite model analysis and static "
        "ESP32-S3 compatibility checks. "
        "Firmware compilation requires the ESP-IDF toolchain (not installed on this host). "
        "Hardware inference has not been performed."
    )

    # ── Session state keys ────────────────────────────────────────────────────
    for _key, _default in [
        ("cf_analysis", None),
        ("cf_tflite_bytes", None),
        ("cf_filename", ""),
        ("cf_compat_report", None),
        ("cf_compat_md", ""),
        ("cf_package_bytes", None),
        ("cf_source", ""),
        ("cf_opt_done", False),
        ("cf_opt_tflite_bytes", None),
        ("cf_opt_analysis", None),
        ("cf_opt_compat_report", None),
        ("cf_opt_compat_md", ""),
        ("cf_opt_package_bytes", None),
    ]:
        if _key not in st.session_state:
            st.session_state[_key] = _default

    # =========================================================================
    # A. Model Selection & Upload
    # =========================================================================
    st.markdown("---")
    st.subheader("A. Model Selection")

    _DSCNN_INT8_PATH = project_root / "models" / "quantized" / "baseline_dscnn_int8.tflite"
    _DSCNN_FP32_PATH = project_root / "models" / "quantized" / "baseline_dscnn_fp32.tflite"

    model_source = st.radio(
        "Choose model source:",
        [
            "Built-in DS-CNN (INT8, 21.77 KB) — demonstration model",
            "Built-in DS-CNN (FP32 TFLite, 28.5 KB) — pre-quantization baseline",
            "Upload a .tflite file",
        ],
        horizontal=False,
        key="cf_model_source_radio",
    )

    upload_error: Optional[str] = None
    tflite_bytes_to_analyze: Optional[bytes] = None
    filename_to_use: str = ""

    if "Built-in DS-CNN (INT8" in model_source:
        if _DSCNN_INT8_PATH.is_file():
            with open(_DSCNN_INT8_PATH, "rb") as _f:
                tflite_bytes_to_analyze = _f.read()
            filename_to_use = "baseline_dscnn_int8.tflite"
            st.success(f"✅ Built-in INT8 model loaded: `{filename_to_use}` "
                       f"({len(tflite_bytes_to_analyze):,} bytes)")
        else:
            upload_error = f"Built-in model not found at `{_DSCNN_INT8_PATH}`. Run quantization first."

    elif "Built-in DS-CNN (FP32" in model_source:
        if _DSCNN_FP32_PATH.is_file():
            with open(_DSCNN_FP32_PATH, "rb") as _f:
                tflite_bytes_to_analyze = _f.read()
            filename_to_use = "baseline_dscnn_fp32.tflite"
            st.success(f"✅ Built-in FP32 TFLite model loaded: `{filename_to_use}` "
                       f"({len(tflite_bytes_to_analyze):,} bytes)")
        else:
            upload_error = f"Built-in FP32 TFLite model not found at `{_DSCNN_FP32_PATH}`."

    else:  # Upload
        _uploaded = st.file_uploader(
            "Upload a TFLite model file (max 50 MB):",
            type=["tflite"],
            key="cf_uploader",
        )
        if _uploaded is not None:
            if _uploaded.size == 0:
                upload_error = "Uploaded file is empty."
            elif _uploaded.size > 50 * 1024 * 1024:
                upload_error = "File exceeds 50 MB limit."
            elif not _uploaded.name.lower().endswith(".tflite"):
                upload_error = f"Unsupported file extension: `{_uploaded.name}`. Only `.tflite` is supported."
            else:
                tflite_bytes_to_analyze = _uploaded.getvalue()
                filename_to_use = _uploaded.name
                st.success(f"✅ Uploaded: `{filename_to_use}` ({len(tflite_bytes_to_analyze):,} bytes)")
                st.caption(
                    "**Supported:** TFLite FlatBuffer (`.tflite`). "
                    "PyTorch `.pt` and ONNX files are not directly supported — "
                    "convert them to TFLite first using the project quantization script."
                )
        else:
            st.info("Upload a `.tflite` file to proceed.")

    if upload_error:
        st.error(f"❌ {upload_error}")
        tflite_bytes_to_analyze = None

    # Trigger (re-)analysis when model selection changes
    _new_key = (model_source, filename_to_use, len(tflite_bytes_to_analyze) if tflite_bytes_to_analyze else 0)
    if tflite_bytes_to_analyze and _new_key != st.session_state.get("cf_source"):
        st.session_state["cf_source"] = _new_key
        st.session_state["cf_analysis"] = None  # reset so analysis reruns
        st.session_state["cf_opt_done"] = False
        st.session_state["cf_package_bytes"] = None

    # =========================================================================
    # B. Model Analysis
    # =========================================================================
    if tflite_bytes_to_analyze is not None:
        st.markdown("---")
        st.subheader("B. Model Analysis")

        if st.session_state["cf_analysis"] is None:
            with st.spinner("Analysing model via TFLite interpreter…"):
                try:
                    _analysis = analyze_tflite_bytes(tflite_bytes_to_analyze, filename_to_use)
                    st.session_state["cf_analysis"] = _analysis
                    st.session_state["cf_tflite_bytes"] = tflite_bytes_to_analyze
                    st.session_state["cf_filename"] = filename_to_use
                except TFLiteAnalysisError as _e:
                    st.error(f"❌ Model analysis failed: {_e}")
                    st.session_state["cf_analysis"] = None

        analysis = st.session_state["cf_analysis"]

        if analysis is not None:
            disp = format_analysis_for_display(analysis)

            _a1, _a2 = st.columns(2)
            with _a1:
                st.metric("File Size", f"{analysis['file_size_kb']:.2f} KB")
                st.metric("Input Shape",  disp.get("Input Shape", "—"))
                st.metric("Input Dtype",  disp.get("Input Dtype", "—"))
                st.metric("Unique Operators", str(disp.get("Operator Count (unique)", "—")))
            with _a2:
                st.metric("FlatBuffer ID", disp.get("FlatBuffer ID", "—"))
                st.metric("Output Shape", disp.get("Output Shape", "—"))
                st.metric("Output Dtype", disp.get("Output Dtype", "—"))
                st.metric("Fully INT8",   disp.get("Fully Integer (INT8 I/O)", "—"))

            with st.expander("🔎 Tensor & Quantization Details", expanded=False):
                inp_t = analysis.get("input_tensors", [])
                oup_t = analysis.get("output_tensors", [])
                if inp_t:
                    st.markdown("**Input Tensor**")
                    st.json(inp_t[0])
                if oup_t:
                    st.markdown("**Output Tensor**")
                    st.json(oup_t[0])
                if analysis.get("unique_operators"):
                    st.markdown("**Unique Operators**")
                    st.write(analysis["unique_operators"])

            with st.expander("🔩 Operator Compatibility", expanded=False):
                supported = analysis.get("tflm_supported_operators", [])
                unsupported = analysis.get("tflm_unsupported_operators", [])
                if supported:
                    st.success(f"TFLM-supported ({len(supported)}): " + ", ".join(supported))
                if unsupported:
                    st.error(f"TFLM-unsupported ({len(unsupported)}): " + ", ".join(unsupported))
                if not supported and not unsupported:
                    st.info("Operator list unavailable (interpreter may not have extracted ops).")

            st.caption(f"SHA-256: `{disp.get('SHA-256', '—')}`")
            st.caption(f"Interpreter analysis: {disp.get('Interpreter Analysis', '—')}")

            # ================================================================
            # C. Target Configuration
            # ================================================================
            st.markdown("---")
            st.subheader("C. Target Configuration")
            st.info(
                "**ESP32-S3** is the configured deployment target. "
                "The profile below represents documented hardware constraints — "
                "**not proof of physical deployment**. No firmware was compiled "
                "and no hardware was connected."
            )

            _c1, _c2, _c3, _c4 = st.columns(4)
            with _c1:
                st.metric("Chip", ESP32S3_PROFILE["chip"])
                st.caption(f"Xtensa LX7 @ {ESP32S3_PROFILE['cpu_freq_mhz']} MHz")
            with _c2:
                st.metric("Total SRAM", f"{ESP32S3_PROFILE['total_sram_kb']} KB")
                st.caption(f"~{ESP32S3_PROFILE['available_sram_kb']} KB available to app")
            with _c3:
                st.metric("SPI Flash", f"{ESP32S3_PROFILE['spi_flash_mb']} MB")
                st.caption("Typical configuration")
            with _c4:
                st.metric("Runtime", "TFLM")
                st.caption("esp-nn vector acceleration")

            # ================================================================
            # D. Optimization Workflow
            # ================================================================
            st.markdown("---")
            st.subheader("D. Optimization Workflow")

            _is_dscnn_int8 = (filename_to_use == "baseline_dscnn_int8.tflite")
            _is_dscnn_fp32 = (filename_to_use == "baseline_dscnn_fp32.tflite")
            _is_already_int8 = analysis.get("is_fully_integer", False)

            if _is_dscnn_int8:
                st.success(
                    "✅ **DS-CNN INT8 — Optimization already complete.**\n\n"
                    "The built-in DS-CNN has been fully optimized: PyTorch FP32 → "
                    "Keras → TFLite INT8 via post-training full-integer quantization "
                    "(150 calibration clips). Measured accuracy: 54.83% (−0.58 pp vs FP32). "
                    "No further compression is applied — the existing artifact is used directly."
                )
                st.session_state["cf_opt_done"] = True
                st.session_state["cf_opt_tflite_bytes"] = tflite_bytes_to_analyze
                st.session_state["cf_opt_analysis"] = analysis

            elif _is_dscnn_fp32:
                st.warning(
                    "ℹ️ **DS-CNN FP32 TFLite selected.** "
                    "This model has float32 I/O. It can be inspected and analyzed but cannot be "
                    "directly re-quantized from a `.tflite` file without the original PyTorch checkpoint. "
                    "Use the **built-in INT8** option for the full demonstration, or select "
                    "DS-CNN INT8 from the model picker above."
                )
                st.info(
                    "**Why not re-quantize here?** Post-training quantization requires a representative "
                    "calibration dataset and the original float model graph. A `.tflite` FlatBuffer "
                    "cannot be directly requantized — it must be converted from the source framework. "
                    "The existing INT8 artifact (`baseline_dscnn_int8.tflite`) was produced via "
                    "the project's `src/quantization/export_int8.py` script."
                )
                st.session_state["cf_opt_done"] = False

            elif _is_already_int8:
                st.info(
                    "ℹ️ **Uploaded model is already fully integer (INT8 I/O).** "
                    "No further quantization is applied. "
                    "Static compatibility analysis and package generation are available below."
                )
                st.session_state["cf_opt_done"] = True
                st.session_state["cf_opt_tflite_bytes"] = tflite_bytes_to_analyze
                st.session_state["cf_opt_analysis"] = analysis

            else:
                st.warning(
                    "⚠️ **Uploaded float TFLite model.** "
                    "Direct requantization of an uploaded `.tflite` is not supported — "
                    "a float TFLite cannot be re-quantized without the original training graph and "
                    "a calibration dataset. Compatibility analysis is still available. "
                    "To obtain an INT8 model, re-export from the source framework."
                )
                st.session_state["cf_opt_done"] = False

            # ================================================================
            # E. Results Comparison (DS-CNN demonstration)
            # ================================================================
            if _is_dscnn_int8 and quant_data:
                st.markdown("---")
                st.subheader("E. Results Comparison (DS-CNN Demonstration)")
                st.markdown(
                    "Comparing original PyTorch FP32 checkpoint to the INT8 TFLite model. "
                    "All values are measured from actual artifacts."
                )

                fp32_m = quant_data["models"]["fp32_pytorch"]
                int8_m = quant_data["models"]["int8_tflite"]
                comp  = quant_data.get("comparison", {})

                _fp32_size = fp32_m.get("file_size_kb", 28.5)
                _int8_size = int8_m.get("file_size_kb", 21.77)
                _ratio = round(_fp32_size / _int8_size, 2) if _int8_size > 0 else 0.0

                _e1, _e2, _e3, _e4 = st.columns(4)
                with _e1:
                    st.metric("Original File Size", f"{_fp32_size:.2f} KB",
                              help="FP32 TFLite (source: measured)")
                with _e2:
                    st.metric("Optimized File Size", f"{_int8_size:.2f} KB",
                              delta=f"-{comp.get('file_size_reduction_pct', 0):.1f}%",
                              delta_color="inverse",
                              help="INT8 TFLite (source: measured)")
                with _e3:
                    st.metric("Compression Ratio", f"{_ratio:.2f}×",
                              help="FP32 size ÷ INT8 size")
                with _e4:
                    _acc_delta = comp.get("accuracy_change_pp", -0.58)
                    st.metric("Accuracy Delta",
                              f"{_acc_delta:+.2f} pp",
                              help=(
                                  f"FP32: {fp32_m.get('test_accuracy',0)*100:.2f}% → "
                                  f"INT8: {int8_m.get('test_accuracy',0)*100:.2f}% "
                                  "(source: measured, 518 test samples)"
                              ))

                comp_rows = [
                    {"Metric": "Format",
                     "Original (FP32)": "PyTorch .pt → TFLite float32",
                     "Optimized (INT8)": "TFLite full-int8",
                     "Source": "Architecture"},
                    {"Metric": "File Size",
                     "Original (FP32)": f"{_fp32_size:.2f} KB",
                     "Optimized (INT8)": f"{_int8_size:.2f} KB",
                     "Source": "Measured"},
                    {"Metric": "Size Reduction",
                     "Original (FP32)": "—",
                     "Optimized (INT8)": f"{comp.get('file_size_reduction_pct',0):.1f}% ({_ratio:.2f}×)",
                     "Source": "Calculated"},
                    {"Metric": "Test Accuracy (518 samples)",
                     "Original (FP32)": f"{fp32_m.get('test_accuracy',0)*100:.2f}%",
                     "Optimized (INT8)": f"{int8_m.get('test_accuracy',0)*100:.2f}%",
                     "Source": "Measured"},
                    {"Metric": "Macro-F1 Score",
                     "Original (FP32)": f"{fp32_m.get('macro_f1',0):.4f}",
                     "Optimized (INT8)": f"{int8_m.get('macro_f1',0):.4f}",
                     "Source": "Measured"},
                    {"Metric": "Parameters",
                     "Original (FP32)": "5,248",
                     "Optimized (INT8)": "5,248",
                     "Source": "Architecture"},
                    {"Metric": "Input Dtype",
                     "Original (FP32)": "float32 [1,1,40,47]",
                     "Optimized (INT8)": "int8 [1,40,47,1] scale=0.4796 zp=54",
                     "Source": "Verified"},
                    {"Metric": "Estimated Tensor Arena",
                     "Original (FP32)": "N/A",
                     "Optimized (INT8)": "30.0 KB (configured; 28.9 KB min)",
                     "Source": "Calculated"},
                    {"Metric": "Firmware Build",
                     "Original (FP32)": "Not performed",
                     "Optimized (INT8)": "Not performed (ESP-IDF not installed)",
                     "Source": "Status"},
                ]
                render_table(pd.DataFrame(comp_rows))

            # ================================================================
            # ESP32-S3 Compatibility Check
            # ================================================================
            st.markdown("---")
            st.subheader("ESP32-S3 Compatibility Check")

            _opt_bytes = st.session_state.get("cf_opt_tflite_bytes") or tflite_bytes_to_analyze
            _opt_analysis = st.session_state.get("cf_opt_analysis") or analysis

            if st.session_state.get("cf_compat_report") is None or st.button(
                "🔍 Run Compatibility Check", key="cf_compat_btn"
            ):
                with st.spinner("Running ESP32-S3 static compatibility checks…"):
                    _compat = check_esp32s3_compatibility(_opt_analysis)
                    _compat_md = generate_compatibility_markdown(
                        _compat, Path(filename_to_use).stem
                    )
                    st.session_state["cf_compat_report"] = _compat
                    st.session_state["cf_compat_md"] = _compat_md

            _compat = st.session_state.get("cf_compat_report")
            if _compat is not None:
                _emoji, _label = _compat.summary_status()
                st.markdown(f"### {_emoji} Overall: {_label}")

                if _compat.errors:
                    for _err in _compat.errors:
                        st.error(f"❌ {_err}")
                if _compat.warnings:
                    for _w in _compat.warnings:
                        st.warning(f"⚠️ {_w}")
                if not _compat.errors and not _compat.warnings:
                    st.success("All checks passed — no errors or warnings.")

                render_table(pd.DataFrame(_compat.checks))

                st.markdown(
                    f"- **Firmware Build:** ⚠️ {_compat.firmware_build_status}  \n"
                    f"- **Hardware Test:** 🛑 {_compat.hardware_test_status}"
                )

            # ================================================================
            # F. ESP32-S3 Deployment Package
            # ================================================================
            st.markdown("---")
            st.subheader("F. ESP32-S3 Deployment Package")

            _compat_for_pkg = st.session_state.get("cf_compat_report")
            _compat_md_for_pkg = st.session_state.get("cf_compat_md", "")

            if _compat_for_pkg is None:
                st.info("Run the compatibility check above before generating the package.")
            else:
                if _compat_for_pkg.errors:
                    st.error(
                        "❌ **Package not generated** — the model failed one or more "
                        "static compatibility checks. Fix the errors listed above first."
                    )
                else:
                    if st.button("📦 Generate Deployment Package", key="cf_build_pkg_btn"):
                        with st.spinner("Building ZIP package…"):
                            try:
                                _json_report = {
                                    **_compat_for_pkg.to_dict(),
                                    "model_analysis": _opt_analysis,
                                }
                                _pkg_bytes = build_deployment_package(
                                    tflite_bytes=_opt_bytes,
                                    tflite_filename=filename_to_use,
                                    analysis=_opt_analysis,
                                    compatibility_report_md=_compat_md_for_pkg,
                                    json_report=_json_report,
                                )
                                st.session_state["cf_package_bytes"] = _pkg_bytes
                            except Exception as _pe:
                                st.error(f"Package generation failed: {_pe}")

                    _pkg = st.session_state.get("cf_package_bytes")
                    if _pkg:
                        _pkg_name = f"{Path(filename_to_use).stem}_esp32s3_deploy.zip"
                        st.success(
                            f"✅ Package ready — {len(_pkg):,} bytes "
                            f"({round(len(_pkg)/1024.0,1):.1f} KB compressed)"
                        )

                        _dl1, _dl2 = st.columns(2)
                        with _dl1:
                            st.download_button(
                                label="⬇️ Download Deployment Package (.zip)",
                                data=_pkg,
                                file_name=_pkg_name,
                                mime="application/zip",
                                use_container_width=True,
                            )
                        with _dl2:
                            if _compat_md_for_pkg:
                                st.download_button(
                                    label="⬇️ Download Compatibility Report (.md)",
                                    data=_compat_md_for_pkg.encode(),
                                    file_name=f"{Path(filename_to_use).stem}_compat.md",
                                    mime="text/markdown",
                                    use_container_width=True,
                                )

                        st.markdown("**Package contents:**")
                        st.markdown(
                            "- `models/{tflite}` — optimized TFLite model  \n"
                            "- `include/model_data.h` — C byte-array header (alignas 8)  \n"
                            "- `include/model_labels.h` — class label string table  \n"
                            "- `main.cpp` — TFLM inference scaffold for ESP32-S3  \n"
                            "- `compatibility_report.md` — static analysis results  \n"
                            "- `deployment_report.json` — machine-readable metrics  \n"
                            "- `README.md` — integration instructions  \n"
                            "- `MANIFEST.json` — artifact sizes and SHA-256 hashes"
                        )

                        st.markdown(
                            "> **Status legend:**  \n"
                            "> ✅ Model export complete  \n"
                            "> ✅ Static compatibility checks passed  \n"
                            "> ⚠️ Firmware build: NOT PERFORMED (ESP-IDF not installed)  \n"
                            "> 🛑 Hardware test: NOT PERFORMED (no physical ESP32-S3 board)"
                        )
