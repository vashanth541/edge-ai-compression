# Model Compression Pipeline for Edge AI Deployment on ESP32

A reproducible open-source pipeline to train, compress, and benchmark lightweight Keyword Spotting (KWS) neural networks for deployment on resource-constrained microcontrollers (specifically the **Espressif ESP32** family, such as ESP32 and ESP32-S3).

---

## 1. Project Overview & Objectives

In Edge AI and TinyML, running deep learning models on low-power microcontrollers requires extreme optimization. The ESP32 microcontroller typically has limited SRAM (~320 KB to 512 KB) and Flash storage (typically 4 MB to 8 MB), with no dedicated GPU or hardware floating-point acceleration on many variants.

This project investigates and implements:
1. **Lightweight Baseline Architecture:** An efficient Keyword Spotting model (e.g., Depthwise Separable CNN / DS-CNN or ResNet-8) trained on the Google Speech Commands dataset (`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`).
2. **Structured Pruning:** Removing redundant convolutional channels/filters to permanently reduce parameter count and computational complexity (FLOPs) without relying on sparse matrix acceleration.
3. **Knowledge Distillation:** Transferring knowledge from a larger teacher network to a compact student network.
4. **Quantization:**
   - Post-Training Quantization (PTQ) to 8-bit integer (`int8`).
   - Quantization-Aware Training (QAT) to preserve accuracy during low-bit representation.
5. **ESP32 Deployment Export:** Exporting models into flatbuffer formats (`.tflite`) and C array headers (`model_data.h`) ready for **TensorFlow Lite for Microcontrollers (TFLite Micro)** and **ESP-NN** kernel acceleration.
6. **Local Web Dashboard (`localhost`):** An interactive local web application to visualize audio preprocessing (spectrograms/MFCCs), trigger compression runs, benchmark latency, and inspect Flash/RAM utilization estimates.
7. **Rigorous Benchmarking:** Comparing accuracy, model size (KB), parameter count, CPU latency, and estimated ESP32 SRAM/Flash constraints. PC measurements are clearly distinguished from microcontroller software estimates.

---

## 2. Target Hardware Constraints (ESP32 Reference)

| Hardware Parameter | ESP32 (Standard Xtensa Dual-Core) | ESP32-S3 (Xtensa with Vector Extensions) |
| :--- | :--- | :--- |
| **Clock Frequency** | Up to 240 MHz | Up to 240 MHz |
| **Internal SRAM** | ~320 KB - 520 KB (usable AI arena ~100–250 KB) | ~512 KB (usable AI arena ~200–350 KB) |
| **Flash Storage** | Typically 4 MB - 8 MB SPI Flash | Typically 8 MB - 16 MB Octal Flash |
| **Vector Instructions** | None (software emulation or ESP-NN assembly) | PIE (Processor Instruction Extension) SIMD for int8 |
| **Target Model Budget** | **Flash:** < 200 KB, **Tensor Arena (RAM):** < 80 KB | **Flash:** < 500 KB, **Tensor Arena (RAM):** < 150 KB |

---

## 3. Host Development Environment

The host environment is verified and configured for CPU-only execution on Windows:
* **Operating System:** Windows 11 Home (64-bit)
* **Processor (CPU):** AMD Ryzen 3 7320U (4 Cores / 8 Threads)
* **Memory:** 7.24 GB visible RAM
* **Acceleration:** CPU-only (lightweight CPU wheels, no CUDA dependency)
* **Python Runtime:** CPython 3.11.15 (64-bit), managed via `uv`

---

## 4. Repository Structure

```text
edge-ai-compression/
├── .gitignore                    # Excludes virtual env, datasets, checkpoints, and caches
├── README.md                     # Project documentation & architectural guide
├── requirements.txt              # Planned CPU-optimized dependencies
├── configs/                      # YAML configuration files (hyperparameters, audio settings)
├── data/
│   ├── raw/                      # Downloaded Speech Commands audio files
│   └── processed/                # Preprocessed MFCC / Spectrogram caches
├── deployment/
│   └── esp32/
│       ├── include/              # Exported C array model headers (e.g., model_data.h)
│       └── src/                  # ESP-IDF / Arduino TFLite Micro runner templates
├── models/
│   ├── checkpoints/              # Floating-point PyTorch weights (.pt)
│   └── exported/                 # Quantized TFLite and ONNX models (.tflite, .onnx)
├── results/
│   ├── plots/                    # Comparison charts (accuracy vs. size, latency trade-offs)
│   └── reports/                  # CSV benchmark logs and evaluation summaries
├── src/                          # Modular Python source package
│   ├── audio/                    # WAV loading, slicing, and MFCC feature extraction
│   ├── benchmarking/             # Parameter counting, FLOPs calculation, latency tests
│   ├── compression/              # Pruning, distillation, PTQ, and QAT routines
│   ├── export/                   # Conversion to TFLite, INT8 calibration, C header export
│   └── models/                   # Neural network architectures (DS-CNN, ResNet-8)
└── tests/                        # Automated unit and smoke tests
    └── test_environment.py       # Stage 1 environment validation test
```

---

## 5. Environment Activation & Test Verification

### Step 1: Verify Virtual Environment
From Windows PowerShell in the project root:
```powershell
.\.venv\Scripts\python.exe --version
```
Expected output: `Python 3.11.15`

### Step 2: Run Automated Test Suite (61 Tests)
Execute all unit and integration tests across the entire repository:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```
Expected output:
```text
======================= 61 passed, 4 warnings in 27s =======================
```

---

## 6. Interactive Reviewer Dashboard

Launch the locally hosted Streamlit demonstration web application:
```powershell
$env:PYTHONPATH="C:\Users\ELCOT\OneDrive\Documents\edge-ai-compression"
.\.venv\Scripts\streamlit.exe run src/dashboard/app.py --server.port 8501
```
Navigate to: **`http://localhost:8501`**

### Dashboard Features:
1. **Overview & Status:** Dataset metadata (4,890 audio files, 12 classes, 0% speaker leakage) and 8-stage progress tracker.
2. **Audio Inference Lab:** Interactive audio loading, waveform plotting, MFCC extraction, and simultaneous live inference across PyTorch FP32 baseline and TensorFlow Lite INT8 quantized models.
3. **Model Comparison:** Side-by-side accuracy, macro-F1, parameter counts, latency benchmarks, and confusion matrices.
4. **Visualizations:** Confusion matrices, loss/accuracy training curves, and latency throughput charts.
5. **Virtual ESP32 Deployment:** Transparent 7-stage workflow display, ESP32-S3 Flash and SRAM budget calculations (52.0 KB total working RAM, leaving 89.8% headroom free), layer-by-layer activation tensor sizing, and full disclosures explaining software-only analytical demonstration on laptop CPU.
6. **Artifacts & Reports:** Interactive report reader and direct downloads for C headers (`model_data.h`, `model_labels.h`), integrity manifests, INT8 `.tflite` model, and metric reports.

---

## 7. Project Roadmap & Stage Execution Status

* [x] **Stage 1: Environment & Project Scaffolding** (Completed)
  * Isolated 64-bit Python 3.11 environment configured via `uv`.
  * Complete directory structure, configuration files, and initial environment smoke tests.
* [x] **Stage 2: Audio Preprocessing & Dataset Pipeline** (Completed)
  * Standardized audio preprocessor (`src/audio/preprocessor.py`): mono conversion, 16 kHz resampling, 1.0s windowing (16000 samples), 40 MFCC / 40 log-Mel filterbank extraction.
  * Formal 12-class label policy: 10 target commands (`down`, `go`, `left`, `no`, `off`, `on`, `right`, `stop`, `up`, `yes`), `_silence_` (ambient noise), and `_unknown_` (other words).
  * Strict speaker-aware split isolation (zero data leakage between train/val/test via SHA-1 speaker hashing).
* [x] **Stage 3: Baseline Neural Network Architecture** (Completed)
  * Ultra-compact Depthwise Separable CNN (`src/models/dscnn.py`): 5,248 trainable parameters (~20.5 KB float32, ~5.1 KB INT8).
  * Operator set restricted strictly to `Conv2D`, `DepthwiseConv2D`, `ReLU`, `BatchNorm2D` (folded), `AdaptiveAvgPool2D`, and `Linear`.
* [x] **Stage 4: Real Speech Commands Training & Evaluation** (Completed)
  * Curated real-audio dataset: 4,890 real WAV files across all 12 classes (149.44 MB storage under `data/raw/speech_commands/`).
  * Speaker-isolated splits: Train (3,936 samples, 829 speakers), Val (436 samples, 98 speakers), Test (518 samples, 102 speakers) with zero speaker overlap.
  * Held-out test evaluation: **55.41% accuracy**, **0.5110 macro-F1**, and **2.29 ms CPU inference latency**.
* [x] **Stage 5: Full INT8 Quantization & Operator Verification** (Completed)
  * Full 8-bit integer post-training quantization (`models/quantized/baseline_dscnn_int8.tflite`).
  * Model size reduced from 102.9 KB (`.pt`) to 21.8 KB (`.tflite`) — 78.85% total reduction; raw weights compressed 4.0× (20.5 KB -> 5.12 KB).
  * Test accuracy: **54.83%** (minimal delta of only **-0.58 percentage points** vs FP32).
  * Operator compatibility: 100% of kernels (`CONV_2D`, `DEPTHWISE_CONV_2D`, `AVERAGE_POOL_2D`, `FULLY_CONNECTED`, `RESHAPE`) verified compatible with TFLM and Espressif `esp-nn`.
* [x] **Stage 6: Interactive Reviewer Dashboard** (Completed)
  * Streamlit local web application providing interactive audio inspection, dual model execution, and visual comparison charts.
* [x] **Stage 7: Virtual ESP32-S3 Embedded Deployment & Live Audio Lab** (Completed)
  * C-array export (`deployment/esp32/include/model_data.h`) with 100% SHA-256 byte-for-byte cryptographic integrity verification.
  * Complete ESP-IDF C++ firmware scaffold under `deployment/esp32/` with FreeRTOS task and TFLM runtime registration.
  * Analytical resource budgeting: 30 KB double-buffered tensor arena (+10.7% safety margin), 52 KB total working SRAM (89.8% headroom on 512 KB ESP32-S3).
* [x] **Stage 8: Validation & Finalization** (Completed)
  * Full regression test suite expanded to 61 tests covering C header integrity, label ordering, quantization metadata, and error handling.
  * Comprehensive [Reproducible Results Report](file:///c:/Users/ELCOT/OneDrive/Documents/edge-ai-compression/results/reports/reproducible_results_report.md) and [Virtual Deployment Resource Report](file:///c:/Users/ELCOT/OneDrive/Documents/edge-ai-compression/results/reports/virtual_deployment_resource_report.md).
  * Step-by-step [Reviewer Demonstration Script](file:///c:/Users/ELCOT/OneDrive/Documents/edge-ai-compression/docs/reviewer_demonstration_script.md).

---

## 8. Benchmark Disclosures & Limitations

* **Host Hardware vs. Embedded:** Inferences executed in the dashboard run on the host **AMD Ryzen 3 7320U CPU**. On-device ESP32-S3 latency is projected from literature (~6–10 ms).
* **Firmware Build Status:** The Espressif ESP-IDF compiler (`idf.py`) is not installed on the laptop host. Firmware project files are verified syntactically, but compilation is formally marked **Template Only / Not Built**.
* **Physical Hardware:** Flashing, power measurements, and physical hardware deployment have **not been performed**.
