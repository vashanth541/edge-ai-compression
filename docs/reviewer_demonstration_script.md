# Reviewer Demonstration Script: Edge AI Model Compression Pipeline

This step-by-step guide walks a reviewer through evaluating the Keyword Spotting (KWS) model compression pipeline, reproducing automated test results, and interacting with the local demonstration dashboard.

---

## 1. Quick Start Prerequisites

* **Operating System:** Windows 10/11 (or Linux/macOS)
* **Python Runtime:** Python 3.11 (64-bit recommended)
* **Hardware:** Standard CPU-only PC (no GPU or microcontroller hardware required)

---

## 2. Step 1 — Run the Automated Test Suite

Open Windows PowerShell in the project root:
```powershell
cd C:\Users\ELCOT\OneDrive\Documents\edge-ai-compression
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

### What to Verify:
* **Test Count:** 61 automated tests pass cleanly with 0 failures (`61 passed`).
* **Test Coverage:**
  * Audio preprocessor (mono, 16 kHz, 40-MFCC extraction).
  * Dataset pipeline and speaker isolation (zero speaker leakage).
  * Model architecture (5,248 parameter DS-CNN).
  * Full INT8 quantization metadata and weights transfer parity.
  * C header export byte-for-byte SHA-256 integrity (`model_data.h`).
  * Class label enum consistency (`model_labels.h`).
  * Analytical SRAM and Flash resource budget calculations.
  * Audio validation, error handling, and dual-model live CPU inference.

---

## 3. Step 2 — Launch the Local Demonstration Dashboard

Launch the interactive Streamlit dashboard:
```powershell
$env:PYTHONPATH="C:\Users\ELCOT\OneDrive\Documents\edge-ai-compression"
.\.venv\Scripts\streamlit.exe run src/dashboard/app.py --server.port 8501
```

Once launched, open your web browser to:
👉 **`http://localhost:8501`**

---

## 4. Step 3 — Reviewer Interactive Walkthrough

### 🎙️ Workflow A: Live Audio Inference Lab
1. In the sidebar radio menu, select **"Audio Inference Lab"**.
2. Under "Select Audio Source", choose **"Bundled Speech Commands Sample (12 classes)"**.
3. Select a keyword (e.g., `yes.wav` or `stop.wav`).
4. **Inspect Audio Features:**
   * Review sample rate (16,000 Hz), duration (1.00 s), and listen to the audio playback.
   * Observe the plotted 1D waveform and the extracted 40 Mel × 47 time frames MFCC spectrogram.
5. **Inspect Live Inference:**
   * Both the **PyTorch FP32 Baseline** and **TensorFlow Lite INT8 Quantized** models execute live on your laptop CPU.
   * Verify model agreement, predicted label, confidence percentages, and top-3 ranking.
   * Notice the laptop CPU latency display (~0.3 ms for INT8 vs ~2.3 ms for FP32).
6. *(Optional)* Select **"Upload Custom WAV File"** and upload your own 1-second audio clip to test input validation and inference.

---

### 📊 Workflow B: Model Comparison & Compression Metrics
1. In the sidebar radio menu, select **"Model Comparison"**.
2. **Review Metrics:**
   * **Test Accuracy:** FP32 Baseline = **55.41%** vs INT8 Quantized = **54.83%** (minimal delta of **-0.58 percentage points**).
   * **Macro-F1:** FP32 = **0.5110** vs INT8 = **0.5179** (+0.0069).
   * **Model Footprint:** Reduced from **102.9 KB** (`.pt` checkpoint) to **21.8 KB** (`.tflite` FlatBuffer) — a **78.85% reduction**.
   * **Raw Weights Compression:** Reduced from **20.5 KB** to **5.12 KB** — an **exact 4.00× compression**.
   * **CPU Speedup:** ~7.6× faster inference on laptop CPU using TFLite integer kernels.
3. Review the per-class precision, recall, and F1 comparison table.

---

### 📈 Workflow C: Visualizations & Analytics
1. In the sidebar radio menu, select **"Visualizations"**.
2. Explore the four interactive tabs:
   * **Compression & Accuracy:** Dual-metric bar charts comparing model sizes and classification accuracy.
   * **Latency & Throughput:** CPU inference benchmarks and sample throughput.
   * **Training Curves:** Loss and accuracy evolution across the 8 training epochs.
   * **Confusion Matrix:** Interactive toggle between FP32 and INT8 12×12 confusion matrices on the 518 held-out test samples.

---

### ⚡ Workflow D: Virtual ESP32-S3 Embedded Deployment
1. In the sidebar radio menu, select **"Virtual ESP32 Deployment"**.
2. **Examine the 7-Stage Pipeline Status:**
   * Stages 1–5 are marked **✅ Verified / Prepared**.
   * Stage 6 (Firmware Build) is marked **⚠️ Template Only** (explaining that ESP-IDF `idf.py` is not installed on the laptop host).
   * Stage 7 (Physical Hardware Execution) is explicitly marked **🛑 Not Performed**.
3. **Inspect Microcontroller Memory Sizing:**
   * **Flash (.rodata):** 21.77 KB (occupies **< 0.55%** of a standard 4 MB Flash chip).
   * **Tensor Arena (RAM):** 30.0 KB (analytical double-buffering requirement: 28.25 KB + **10.7% engineering safety margin**).
   * **Total Working SRAM:** 52.0 KB (occupies **10.16%** of 512 KB internal SRAM).
   * **Available SRAM Headroom:** **460 KB (89.84% free)** for networking and application tasks.
4. **Inspect Layer-by-Layer Sizing Table:**
   * View intermediate tensor shapes and peak memory consumers (Convolution blocks @ 13.12 KB each).

---

### 📁 Workflow E: Artifacts & Reports Explorer
1. In the sidebar radio menu, select **"Artifacts & Reports"**.
2. **Inventory Table:** Confirm that all 18 tracked project files exist on disk with valid file sizes.
3. **Download Artifacts:**
   * Click download buttons to inspect `model_data.h`, `model_labels.h`, `model_export_manifest.json`, `baseline_dscnn_int8.tflite`, and JSON metric reports.
4. **Interactive Report Viewer:**
   * Switch between the Virtual Deployment, Operator Compatibility, Quantization Comparison, and Baseline reports rendered live from repository Markdown files.

---

## 5. Reviewer Summary & Verification Checklist

- [x] All 61 automated tests pass (`pytest tests/ -v`).
- [x] INT8 TFLite model is 100% full-integer quantized (int8 weights/activations, int32 biases).
- [x] `model_data.h` matches `baseline_dscnn_int8.tflite` byte-for-byte (SHA-256 verified).
- [x] Label mappings are consistent across model outputs, C headers, and dataset config.
- [x] ESP32-S3 memory allocations are analytically calculated with explicit safety margins.
- [x] Laptop CPU latency is clearly distinguished from embedded microcontroller projections.
- [x] Physical hardware flashing is transparently disclosed as "Not performed".
