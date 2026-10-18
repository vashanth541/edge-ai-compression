# Evidence-Based Technical Audit Report: Edge AI Model Compression Pipeline

**Project:** 12-Class Keyword Spotting (KWS) Depthwise Separable CNN (DS-CNN)  
**Target Hardware:** Espressif ESP32-S3 (Xtensa LX7 Dual-Core @ 240 MHz)  
**Audit Stage:** Stage 9 — Technical Verification & Evidence Audit  
**Host Environment:** Windows 11 Home (64-bit), AMD Ryzen 3 7320U CPU, Python 3.11.15  
**Audit Date:** 2026-10-18  
**Automated Test Status:** 61 passed, 0 failed, 4 warnings in 43.73s (`pytest tests/ -v`)  

---

## 1. Executive Summary

This independent technical audit was conducted to rigorously verify all engineering claims, numerical metrics, model artifacts, firmware templates, and documentation across the Edge AI Compression Pipeline. Every claim was evaluated against raw code, binary files, execution logs, and automated tests.

### Key Audit Conclusions:
1. **Model Parameter Count (Verified):** The DS-CNN architecture instantiates exactly **5,248 trainable parameters** (0 non-trainable, 513 BatchNorm buffer elements), matching the documented claim 100%.
2. **Computational Complexity (Verified):** The model requires **2,110,416 MACs (~2.11 MMACs)** and **4,220,832 FLOPs** (under the standard convention $1\text{ MAC} = 2\text{ FLOPs}$) for an input tensor of shape `[1, 1, 40, 47]`.
3. **Storage & Quantization (Verified):** Raw FP32 weights occupy **20,992 bytes (20.50 KB)**, while the `.pt` checkpoint file is **105,396 bytes (102.93 KB)** due to Adam optimizer state and metadata. Full integer INT8 quantization compresses the model to a **22,296-byte (21.77 KB)** FlatBuffer containing 28 `int8` and 6 `int32` tensors (0 `float32` tensors), achieving a **78.85% total file reduction** and **4.0× weight compression**.
4. **Accuracy & Metric Parity (Verified):** On the exact same 518 held-out test clips, FP32 baseline achieved **55.41% accuracy (287/518)** and **0.5110 macro-F1**; INT8 quantized model achieved **54.83% accuracy (284/518)** and **0.5179 macro-F1** (a minimal delta of **-0.58 percentage points**).
5. **C Header Cryptographic Integrity (Verified):** `deployment/esp32/include/model_data.h` matches `baseline_dscnn_int8.tflite` byte-for-byte with identical SHA-256 hash `4753e7911e93f23a90d5bf19038eedd67312d381123e83d89bcb16c0cd5f6fe7`.
6. **Hardware & Firmware Disclosures (Critical):** ESP-IDF (`idf.py`) is **not installed** on this host laptop. The firmware in `deployment/esp32/` is a **syntactically validated template that has not been compiled**. SRAM requirements (52.0 KB) and tensor arena (30.0 KB) are **calculated estimates**, and ESP32-S3 latency (6–10 ms) is a **literature-based projection**. No physical ESP32-S3 execution, microphone capture, or power measurement was performed.

---

## 2. Model Architecture and Parameter Verification

### 2.1 Layer-by-Layer Architecture & Tensor Dimensions

An input tensor of shape `[1, 1, 40, 47]` (Batch, Channels, MFCCs, Time Frames) traverses the network as follows:

| Layer / Block | Component Description | Output Tensor Dimensions | Trainable Parameters | Buffer Elements |
| :--- | :--- | :---: | :---: | :---: |
| **Input** | Audio MFCC Spectrogram | `[1, 1, 40, 47]` | 0 | 0 |
| **Stem Conv2D** | `Conv2D(1, 28, k=(3,3), s=(2,2), p=(1,1), bias=False)` | `[1, 28, 20, 24]` | 252 | 0 |
| **Stem BN + ReLU** | `BatchNorm2d(28) + ReLU()` | `[1, 28, 20, 24]` | 56 | 57 |
| **DS Block 0 (DW)**| `Conv2d(28, 28, k=(3,3), s=(1,1), p=(1,1), groups=28, bias=False) + BN + ReLU` | `[1, 28, 20, 24]` | 308 (252 + 56) | 57 |
| **DS Block 0 (PW)**| `Conv2d(28, 28, k=(1,1), s=(1,1), bias=False) + BN + ReLU` | `[1, 28, 20, 24]` | 840 (784 + 56) | 57 |
| **DS Block 1 (DW)**| `Conv2d(28, 28, k=(3,3), s=(1,1), p=(1,1), groups=28, bias=False) + BN + ReLU` | `[1, 28, 20, 24]` | 308 (252 + 56) | 57 |
| **DS Block 1 (PW)**| `Conv2d(28, 28, k=(1,1), s=(1,1), bias=False) + BN + ReLU` | `[1, 28, 20, 24]` | 840 (784 + 56) | 57 |
| **DS Block 2 (DW)**| `Conv2d(28, 28, k=(3,3), s=(1,1), p=(1,1), groups=28, bias=False) + BN + ReLU` | `[1, 28, 20, 24]` | 308 (252 + 56) | 57 |
| **DS Block 2 (PW)**| `Conv2d(28, 28, k=(1,1), s=(1,1), bias=False) + BN + ReLU` | `[1, 28, 20, 24]` | 840 (784 + 56) | 57 |
| **DS Block 3 (DW)**| `Conv2d(28, 28, k=(3,3), s=(1,1), p=(1,1), groups=28, bias=False) + BN + ReLU` | `[1, 28, 20, 24]` | 308 (252 + 56) | 57 |
| **DS Block 3 (PW)**| `Conv2d(28, 28, k=(1,1), s=(1,1), bias=False) + BN + ReLU` | `[1, 28, 20, 24]` | 840 (784 + 56) | 57 |
| **Global Pool** | `AdaptiveAvgPool2d((1, 1))` | `[1, 28, 1, 1]` | 0 | 0 |
| **Flatten** | `torch.flatten(x, 1)` | `[1, 28]` | 0 | 0 |
| **Dropout** | `Dropout(p=0.1)` | `[1, 28]` | 0 | 0 |
| **Classifier FC**| `Linear(in=28, out=12, bias=True)` | `[1, 12]` | 348 (336 + 12) | 0 |
| **TOTAL** | — | — | **5,248** | **513** |

* **Total Parameters:** 5,248
* **Trainable Parameters:** 5,248
* **Non-trainable Parameters:** 0
* **Non-parameter Buffers:** 513 elements (9 BatchNorm layers $\times$ [28 running mean + 28 running var + 1 num batches tracked] = 513 elements).

### 2.2 Computational Complexity (MACs and FLOPs)

* **Stem Conv2D:** $20 \times 24 \times (1 \times 3 \times 3) \times 28 = 120,960\text{ MACs}$
* **DS Blocks (4 Blocks):**
  * Depthwise Conv: $20 \times 24 \times (1 \times 3 \times 3) \times 28 = 120,960\text{ MACs}$ per block
  * Pointwise Conv: $20 \times 24 \times (28 \times 1 \times 1) \times 28 = 376,320\text{ MACs}$ per block
  * Per Block: $120,960 + 376,320 = 497,280\text{ MACs}$
  * 4 Blocks Total: $4 \times 497,280 = 1,989,120\text{ MACs}$
* **Classifier FC:** $28 \times 12 = 336\text{ MACs}$
* **Total Multiply-Accumulates (MACs):** **2,110,416 MACs (~2.11 MMACs)**
* **Total Floating-Point Operations (FLOPs):** Under standard machine learning accounting where $1\text{ MAC} = 1\text{ multiplication} + 1\text{ addition} = 2\text{ FLOPs}$:
  $$\text{Total FLOPs} = 2 \times 2,110,416 = \mathbf{4,220,832\text{ FLOPs}}\text{ (~4.22 MFLOPs)}$$
  *(Note: Including GAP additions and FC bias addition adds 13,452 operations for 4,234,284 FLOPs).*

### 2.3 Checkpoint File Size vs. Raw Weight Storage

* **Raw FP32 Weight Payload:** $5,248\text{ parameters} \times 4\text{ bytes/float32} = \mathbf{20,992\text{ bytes (20.50 KB)}}$.
* **Checkpoint File on Disk (`baseline_dscnn.pt`):** $\mathbf{105,396\text{ bytes (102.93 KB)}}$.
* **Overhead Breakdown (84,404 bytes):**
  * Adam optimizer state dict (first and second momentum buffers `exp_avg` and `exp_avg_sq` for 5,248 parameters: $2 \times 5,248 \times 4 = 41,984\text{ bytes}$).
  * BatchNorm tracking buffers (513 floats $\approx 2,052\text{ bytes}$).
  * Model architecture config dict, training metadata, and PyTorch ZIP/pickle container overhead.
* **Finding:** Reports and presentations must clearly distinguish total checkpoint file size (102.93 KB) from raw parameter storage (20.50 KB) to prevent reviewer confusion.

---

## 3. Audio Preprocessing Pipeline Verification

### 3.1 Preprocessing Parameter Audit

| Parameter | Configured Value | Verification Method | Status |
| :--- | :---: | :--- | :---: |
| **Sample Rate** | 16,000 Hz | Checked `configs/audio_config.yaml` & `preprocessor.py` | Verified |
| **Channels** | 1 (Mono) | Validated `to_mono()` downmix logic | Verified |
| **Duration / Samples** | 1.0 s / 16,000 samples | Verified `pad_or_crop()` logic | Verified |
| **Window Length (`win_length`)** | 640 samples (40.0 ms) | Calculated from `window_size_ms` | Verified |
| **Hop Length (`hop_length`)** | 320 samples (20.0 ms) | Calculated from `window_stride_ms` | Verified |
| **FFT Size (`n_fft`)** | 1,024 | Verified STFT parameters | Verified |
| **Window Centering** | `center=False` | Verified in `torchaudio.transforms.MFCC` | Verified |
| **Mel Filterbanks (`n_mels`)**| 40 (`f_min=20Hz`, `f_max=4000Hz`) | Validated Mel filter bank definition | Verified |
| **MFCC Coefficients (`n_mfcc`)**| 40 coefficients | Validated output tensor dimension | Verified |
| **Time Frame Count** | 47 frames | $(16,000 - 1,024) // 320 + 1 = 47$ | Verified |

### 3.2 Observed Output Tensor Properties

Running `AudioPreprocessor.process_file()` on genuine dataset WAV files produced:
* **Shape:** Exactly `torch.Size([1, 40, 47])`
* **Dtype:** `torch.float32`
* **Value Integrity:** Zero NaNs, zero Infs; all values finite float32.

### 3.3 Pipeline Compatibility

Cross-inspection of `src/audio/dataset.py`, `src/train.py`, `src/evaluate.py`, `src/quantization/export_int8.py`, `src/quantization/evaluate_quantized.py`, `src/deployment/inference_engine.py`, and `src/dashboard/app.py` confirmed that **all execution paths use identical preprocessing** via the shared `AudioPreprocessor` class.

---

## 4. Dataset Integrity and Evaluation Verification

### 4.1 Dataset Composition & 12-Class Label Ordering

The dataset located at `data/raw/speech_commands` contains **4,890 genuine audio files** organized into 12 subdirectories:

| Class Index | Label Name | File Count | Label Type / Source |
| :---: | :--- | :---: | :--- |
| 0 | `down` | 406 | Target Command |
| 1 | `go` | 402 | Target Command |
| 2 | `left` | 412 | Target Command |
| 3 | `no` | 405 | Target Command |
| 4 | `off` | 402 | Target Command |
| 5 | `on` | 396 | Target Command |
| 6 | `right` | 396 | Target Command |
| 7 | `stop` | 411 | Target Command |
| 8 | `up` | 425 | Target Command |
| 9 | `yes` | 419 | Target Command |
| 10 | `_silence_` | 408 | Ambient background noise slices |
| 11 | `_unknown_` | 408 | Non-target words (`backward`, `bed`, `cat`, etc.) |
| **Total** | — | **4,890** | — |

* **Label Ordering Parity:** The C header `model_labels.h` matches `DatasetConfig.get_class_to_idx()` with 100% enum and array alignment.

### 4.2 Speaker Isolation & Leakage Verification

* **Speaker Extraction:** `get_speaker_id()` parses `{speaker_id}_nohash_{clip}.wav` stems.
* **Deterministic Partitioning:**
  $$\text{bucket} = \text{int}(\text{SHA1}(\text{speaker\_id})\text{.hexdigest}(), 16) \pmod{100}$$
  * Bucket $< 10 \implies$ Validation split
  * Bucket $10 \dots 19 \implies$ Test split
  * Bucket $\ge 20 \implies$ Training split
* **Audit Findings:**
  * **Train Split:** 3,936 clips (829 unique speaker IDs)
  * **Val Split:** 436 clips (98 unique speaker IDs)
  * **Test Split:** 518 clips (102 unique speaker IDs)
  * **Overlap:** $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$. **Zero speaker leakage (0.0%)**.
* **Critical Clarification:** SHA-1 is applied to the speaker ID string as a deterministic pseudo-random bucketing function, not as a biometric signature or proof of speaker identity.

### 4.3 Evaluation Re-Run & Metric Parity

Re-running evaluation across all 518 held-out test samples confirmed exact mathematical reproduction:

| Metric | Documented (Report / JSON) | Re-Evaluated Live | Parity Status |
| :--- | :---: | :---: | :---: |
| **FP32 Test Accuracy** | 55.41% (287 / 518) | 55.41% (287 / 518) | **100% Match** |
| **FP32 Macro-F1** | 0.5110 | 0.5110 | **100% Match** |
| **INT8 Test Accuracy** | 54.83% (284 / 518) | 54.83% (284 / 518) | **100% Match** |
| **INT8 Macro-F1** | 0.5179 | 0.5179 | **100% Match** |
| **Accuracy Delta** | -0.58 percentage points | -0.58 percentage points | **100% Match** |
| **Macro-F1 Delta** | +0.0069 | +0.0069 | **100% Match** |
| **Test Clips in Calibration?** | No | No (Verified `split='train'`) | **Confirmed Isolated** |

---

## 5. INT8 Quantization and Artifact Cryptographic Integrity

### 5.1 TFLite Model Inspection

* **File Path:** `models/quantized/baseline_dscnn_int8.tflite`
* **File Size:** **22,296 bytes (21.77 KB)**
* **Input Tensor:** Name `serving_default_input:0`, Shape `[1, 40, 47, 1]`, Dtype `int8`, Scale `0.479631`, Zero Point `54`.
* **Output Tensor:** Name `StatefulPartitionedCall_1:0`, Shape `[1, 12]`, Dtype `int8`, Scale `0.100944`, Zero Point `82`.
* **Tensor Breakdown:**
  * 28 `int8` tensors (inputs, weights, intermediate activations, outputs).
  * 6 `int32` tensors (biases for Conv2D, DepthwiseConv2D, and Dense layers).
  * 0 `float32` tensors.
* **Full Integer Quantization Verdict:** Confirmed. Zero floating-point fallbacks.

### 5.2 C Header Cryptographic Verification

* **C Header File:** `deployment/esp32/include/model_data.h`
* **Extracted Byte Count:** 22,296 bytes
* **Target `.tflite` File Size:** 22,296 bytes
* **TFLite SHA-256:** `4753e7911e93f23a90d5bf19038eedd67312d381123e83d89bcb16c0cd5f6fe7`
* **Header Hex Array SHA-256:** `4753e7911e93f23a90d5bf19038eedd67312d381123e83d89bcb16c0cd5f6fe7`
* **Cryptographic Verdict:** **100% exact byte match**.

---

## 6. ESP32-S3 Virtual Deployment & Hardware Claims Audit

### 6.1 Status Taxonomy & Evidence Labels

Every deployment claim has been strictly classified under the required audit taxonomy:

| Item / Claim | Documented Claim | Audit Evidence / Verification | Evidence Label |
| :--- | :---: | :--- | :--- |
| **Model Parameter Count** | 5,248 parameters | Evaluated in `src/models/dscnn.py` & `test_model_dscnn.py` | **1. Verified by source code** |
| **Quantized Model Size** | 21.77 KB (22,296 B) | File size & byte check on `baseline_dscnn_int8.tflite` | **1. Verified by source code** |
| **C Array Header Match** | 100% SHA-256 match | Computed SHA-256 on `model_data.h` vs `.tflite` | **1. Verified by source code** |
| **Automated Tests** | 61 passing tests | Executed `pytest tests/ -v` (61 passed in 43.73s) | **1. Verified by automated test** |
| **Tensor Arena Size** | 30.00 KB | Calculated: peak activations $26.88\text{ KB} + 2.0\text{ KB metadata} = 28.25\text{ KB}$ | **2. Calculated estimate** |
| **Working SRAM Memory** | 52.00 KB | Calculated: $30\text{ KB arena} + 16\text{ KB stack} + 2\text{ KB DMA} + 4\text{ KB MFCC}$ | **2. Calculated estimate** |
| **Flash Memory Occupancy** | 0.54% of 4 MB Flash | Calculated: $22,296\text{ B} / 4,194,304\text{ B} = 0.5316\%$ | **2. Calculated estimate** |
| **Projected ESP32-S3 Latency**| ~6–10 ms @ 240 MHz | Extrapolated from published literature for 2.11 MMACs on Xtensa PIE | **3. Literature-based projection** |
| **ESP-IDF Firmware Build** | `deployment/esp32/` | `idf.py` is **not installed**; compilation **not executed** | **NOT BUILT (Template Only)** |
| **Physical ESP32 Flashing** | Physical execution | No physical microcontroller connected; no current/power measurement | **NOT PERFORMED** |

### 6.2 Explicit Negative Declarations for Viva & Reviewers:
1. **No Firmware Binary:** No `.bin` or `.elf` executable was compiled because the Espressif ESP-IDF compiler (`idf.py`) is not installed on this host machine.
2. **No Physical Microcontroller Execution:** No physical ESP32 or ESP32-S3 chip was flashed or executed.
3. **No Live Microphone Capture on Embedded Silicon:** No live hardware I2S microphone (e.g. INMP441) was evaluated on a microcontroller.
4. **No Electrical Power Measurements:** No current (mA) or power (mW) measurements were taken on physical silicon.
5. **Laptop Latency vs. Embedded Latency:** The measured 0.30 ms latency was obtained on an **AMD Ryzen 3 7320U laptop processor** running Windows 11. It must never be presented as an embedded microcontroller measurement.

---

## 7. Dashboard and Test Suite Verification

### 7.1 Automated Regression Test Suite

* **Command Executed:** `.\.venv\Scripts\python.exe -m pytest tests/ -v`
* **Test Framework:** `pytest 9.1.1`, `pluggy 1.7.0`
* **Pass Count:** **61 passed**
* **Failures:** **0 failed**
* **Skips:** **0 skipped**
* **Warnings:** **4 warnings** (Deprecation notices from `tensorflow.lite.python.interpreter` regarding future LiteRT migration).
* **Execution Duration:** 43.73 seconds.

### 7.2 Dashboard Verification & Downloadable Artifact Integrity

All 14 downloadable artifacts registered in `src/dashboard/data_loader.py` were verified to exist on disk and match their documented file sizes and MIME types.

---

## 8. Documentation Reconciliation & Actionable Proposals

### 8.1 Discrepancy Catalog

1. **Per-Class Metrics Mismatch in `reproducible_results_report.md` (Medium Severity):**
   * *Observation:* Table 4 in `reproducible_results_report.md` contains per-class precision/recall/F1 figures for `stop`, `up`, `go`, and `left` that differ from the exact JSON values in `quantization_comparison.json`, despite the overall macro average (0.511 / 0.518) matching.
   * *Proposed Correction:* Update Table 4 in `reproducible_results_report.md` with the verified numbers from `quantization_comparison.json`.
2. **Calibration Sample Count in Presentation Slide 7 (Low Severity):**
   * *Observation:* Slide 7 of `scratch/generate_presentation.py` states "Calibrated using 100 genuine training audio clips", whereas `configs/quantization_config.yaml` specifies `representative_samples_count: 150`.
   * *Proposed Correction:* Update Slide 7 to state "150 genuine training audio clips".
3. **Vocabulary Terminology regarding SHA-1 (Low Severity):**
   * *Observation:* Certain sections describe SHA-1 as ensuring speaker isolation, which could be misconstrued as biometric speaker identification.
   * *Proposed Correction:* Explicitly state that SHA-1 is applied as a deterministic pseudo-random hash to the speaker ID string extracted from filenames.

---

## 9. Recommended Next Action

1. **Present with Strict Transparency:** During project viva and demonstrations, present the project as a **rigorous model compression and embedded deployment pipeline with software-verified virtual deployment scaffolding**.
2. **Keep Original Artifacts Unchanged:** As instructed, all original reports and model checkpoints have been preserved without overwriting.
3. **If Desired:** Apply the proposed minor documentation reconciliations to `reproducible_results_report.md` and `generate_presentation.py` upon explicit approval.
