# Reproducible Results Report: Edge AI Model Compression Pipeline

**Project:** 12-Class Keyword Spotting (KWS) Depthwise Separable CNN (DS-CNN)  
**Target:** Espressif ESP32-S3 (Virtual Analytical Deployment)  
**Host Environment:** Windows 11 Home (64-bit), AMD Ryzen 3 7320U CPU, Python 3.11.15  
**Verification Date:** 2026-10-17  
**Test Suite Status:** 61 passed, 0 failed, 4 warnings (`pytest tests/ -v`)

---

## 1. Executive Summary

This report establishes the verified, reproducible benchmarks for the Edge AI Model Compression Pipeline. The objective was to train a compact Keyword Spotting model on the Google Speech Commands dataset, apply full 8-bit integer post-training quantization (PTQ), export a verified C-array header for microcontroller deployment, and evaluate the trade-offs across accuracy, model footprint, and execution latency.

All results reported below have been reproduced on the actual project artifacts and verified with automated regression tests.

---

## 2. Model Architecture & Compression Summary

| Metric | Floating-Point Baseline (FP32) | Post-Training Quantized (INT8) | Relative Delta / Compression | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Model Format** | PyTorch State Dict (`.pt`) | TensorFlow Lite FlatBuffer (`.tflite`) | Format migration | Verified |
| **File Path** | `models/checkpoints/baseline_dscnn.pt` | `models/quantized/baseline_dscnn_int8.tflite` | — | Verified on disk |
| **Total Parameters** | 5,248 parameters | 5,248 parameters | 1:1 parameter parity | Verified |
| **Model File Size** | 105,396 bytes (102.93 KB) | 22,296 bytes (21.77 KB) | **-78.85%** (4.73× reduction) | Exact file size |
| **Raw Weights Size** | 20,992 bytes (20.50 KB) | 5,248 bytes (5.125 KB) | **-75.00%** (**4.00× reduction**) | Exact byte count |
| **Test Accuracy** | **55.41%** (287 / 518 correct) | **54.83%** (284 / 518 correct) | **-0.58 percentage points** | Evaluated on identical test set |
| **Macro-F1 Score** | 0.5110 | 0.5179 | **+0.0069** (+0.69 pp) | Evaluated on identical test set |
| **Laptop CPU Latency** | 2.29 ms ± 0.96 ms | 0.30 ms ± 0.08 ms | **7.59× speedup** on laptop CPU | 10 warm-up, 100 timed runs |
| **Laptop CPU Throughput** | 436.5 samples/sec | 3,312.3 samples/sec | 7.59× throughput increase | Laptop CPU (XNNPACK) |
| **Target Runtime** | PyTorch 2.14.1 (CPU) | TFLite / TFLite Micro | 100% Integer Ops | Verified compatible |

---

## 3. Dataset Configuration & Evaluation Methodology

* **Dataset:** Google Speech Commands v2 curated subset.
* **Total Audio Clips:** 4,890 verified `.wav` files (149.44 MB).
* **Audio Input Format:** 16,000 Hz, 1-channel mono, exactly 1.0 second (16,000 samples).
* **Input Feature:** 40-coefficient Mel-Frequency Cepstral Coefficients (MFCC) over 47 time frames (shape `[1, 40, 47]`).
* **12 Target Classes:**
  * 10 core keywords: `down`, `go`, `left`, `no`, `off`, `on`, `right`, `stop`, `up`, `yes`
  * 2 special classes: `_silence_` (background ambient noise), `_unknown_` (other speech)
* **Split Policy & Speaker Isolation:**
  * **Training Split:** 3,936 clips (829 unique speakers)
  * **Validation Split:** 436 clips (98 unique speakers)
  * **Held-Out Test Split:** 518 clips (102 unique speakers)
  * **Leakage Prevention:** Deterministic SHA-1 speaker hashing ensures zero speaker overlap across splits (0.0% leakage).
* **Fair Comparison Guarantee:** Both the FP32 baseline and INT8 quantized models were evaluated on the **exact same pre-extracted 518 test feature tensors**, using the exact same class-to-index mapping (0 to 11).

---

## 4. Per-Class Precision, Recall, and F1 Comparison

| Class Label | Index | Support | FP32 Precision | FP32 Recall | FP32 F1 | INT8 Precision | INT8 Recall | INT8 F1 | F1 Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `down` | 0 | 40 | 0.000 | 0.000 | 0.000 | 0.231 | 0.075 | 0.113 | +0.113 |
| `go` | 1 | 49 | 0.619 | 0.531 | 0.571 | 0.600 | 0.612 | 0.606 | +0.035 |
| `left` | 2 | 44 | 0.500 | 0.159 | 0.241 | 0.560 | 0.318 | 0.406 | +0.165 |
| `no` | 3 | 43 | 0.393 | 0.814 | 0.530 | 0.436 | 0.791 | 0.562 | +0.032 |
| `off` | 4 | 45 | 0.453 | 0.644 | 0.532 | 0.429 | 0.667 | 0.522 | -0.010 |
| `on` | 5 | 44 | 0.534 | 0.705 | 0.608 | 0.511 | 0.545 | 0.527 | -0.081 |
| `right` | 6 | 32 | 0.509 | 0.875 | 0.644 | 0.547 | 0.906 | 0.682 | +0.038 |
| `stop` | 7 | 49 | 0.545 | 0.490 | 0.516 | 0.490 | 0.490 | 0.490 | -0.026 |
| `up` | 8 | 43 | 0.345 | 0.233 | 0.278 | 0.294 | 0.116 | 0.167 | -0.111 |
| `yes` | 9 | 50 | 0.737 | 0.840 | 0.785 | 0.804 | 0.740 | 0.771 | -0.014 |
| `_silence_` | 10 | 42 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| `_unknown_` | 11 | 37 | 0.542 | 0.351 | 0.426 | 0.429 | 0.324 | 0.369 | -0.057 |
| **Macro Average** | — | **518** | **0.511** | **0.540** | **0.511** | **0.512** | **0.546** | **0.518** | **+0.007** |

---

## 5. C-Header Cryptographic Verification

* **Export Tool:** [`src/export/export_c_header.py`](file:///c:/Users/ELCOT/OneDrive/Documents/edge-ai-compression/src/export/export_c_header.py)
* **Source FlatBuffer:** `models/quantized/baseline_dscnn_int8.tflite`
* **Target Header:** `deployment/esp32/include/model_data.h`
* **Target Label Header:** `deployment/esp32/include/model_labels.h`
* **Manifest:** `deployment/esp32/model_export_manifest.json`

### Verification Results
1. **Byte Count:** Exact match at 22,296 bytes.
2. **SHA-256 Hash:** `4753e7911e93f23a90d5bf19038eedd67312d381123e83d89bcb16c0cd5f6fe7` (100% cryptographic parity).
3. **Reconstructed Binary Identity:** Rebuilding the binary array byte-by-byte from hex literals in `model_data.h` yields zero differing bytes compared to `baseline_dscnn_int8.tflite`.
4. **Alignment:** The C array is declared with `TFLM_ALIGN alignas(8)` to ensure 64-bit alignment required for SIMD execution on microcontrollers.
5. **Class Mapping:** Verified 100% consistency across `model_labels.h` enum, C string array, and `DatasetConfig.get_idx_to_class()`.

---

## 6. Host Hardware vs. Microcontroller Disclosures

> [!IMPORTANT]
> **CRITICAL REVIEWER NOTICE: BENCHMARK DEVICE DISCLOSURE**
> 
> * **Latency & Throughput:** The inference benchmarks reported above (0.30 ms per sample) were collected on an **AMD Ryzen 3 7320U laptop processor** running Windows 11.
> * **Microcontroller Projection:** On an ESP32-S3 @ 240 MHz using `esp-nn` vector assembly kernels, single-inference latency is projected from published literature to be approximately **6–10 ms**.
> * **Toolchain Status:** The Espressif ESP-IDF compiler toolchain (`idf.py`) is not installed on this host machine. The firmware project under `deployment/esp32/` is complete and verified syntactically, but compilation is formally marked **Template Only / Not Built**.
> * **Hardware Deployment:** Physical flashing, current draw, battery life, and on-silicon latency measurements have **NOT been performed**.
