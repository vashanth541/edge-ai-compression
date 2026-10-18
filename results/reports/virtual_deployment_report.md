# Stage 7: ESP32-S3 Virtual Deployment & Resource Analysis Report

**Target Hardware:** Espressif ESP32-S3 (Dual-Core Xtensa LX7 @ 240 MHz, 512 KB internal SRAM, 4 MB SPI Flash)  
**Model Architecture:** DS-CNN (5,248 parameters, full integer INT8 quantization)  
**Host Environment:** Windows 11 (64-bit), CPU-only (Laptop verification)  
**Evaluation Status:** Software-Only Virtual Deployment Analysis (No physical hardware measurement)  

---

## 1. Model Storage & Binary Footprint

| Component | Size (Bytes) | Size (KB) | Ratio / Notes |
| :--- | :---: | :---: | :--- |
| **TFLite FlatBuffer Binary** | 22,296 B | 21.77 KB | Source model on disk |
| **Raw Parameter Weights Payload** | 5,248 B | 5.12 KB | 5,248 parameters × 1 Byte (INT8) |
| **FlatBuffer Metadata & Headers** | 17,048 B | 16.65 KB | Tensor descriptors, scales, zero-points |
| **Compiled `.rodata` Flash Payload** | 22,296 B | 21.77 KB | 8-byte aligned memory payload |
| **C Header Source Code Text** | 144,837 B | 141.44 KB | `model_data.h` ASCII representation |
| **Model SHA-256 Checksum** | `4753e7911e93f23a90d5bf19038eedd67312d381123e83d89bcb16c0cd5f6fe7` | — | Byte-for-byte verified |

---

## 2. Flash Memory Occupancy (ESP32-S3 4 MB Flash)

- **Total Chip Flash:** 4.0 MB (4,194,304 bytes)
- **Standard App Partition:** 1.5 MB (1,572,864 bytes)
- **Model Occupancy (App Partition):** **1.42%** (21.77 KB used)
- **Model Occupancy (Total Flash):** **0.532%**
- **Available Flash Headroom:** **98.58%** remaining for application logic, Wi-Fi stacks, and OTA bootloader.

---

## 3. Layer-by-Layer Activation Analysis & Tensor Arena

The Tensor Arena cannot be guessed from model file size alone. It is determined by the peak consecutive activation buffers in the computation graph:

| Layer / Operator | Output Tensor Shape | Activation Buffer Size | Data Type |
| :--- | :---: | :---: | :---: |
| Input Spectrogram | `[1, 40, 47, 1]` | 1,880 Bytes | `int8` |
| Conv2D (28 filters, stride 2) | `[1, 20, 24, 28]` | **13,440 Bytes** | `int8` |
| Depthwise Conv Block 1 | `[1, 20, 24, 28]` | **13,440 Bytes** | `int8` |
| Pointwise Conv Block 1 | `[1, 20, 24, 28]` | **13,440 Bytes** | `int8` |
| Depthwise Conv Block 2 | `[1, 20, 24, 28]` | **13,440 Bytes** | `int8` |
| Pointwise Conv Block 2 | `[1, 20, 24, 28]` | **13,440 Bytes** | `int8` |
| Global Average Pooling | `[1, 28]` | 28 Bytes | `int8` |
| Fully Connected Output | `[1, 12]` | 12 Bytes | `int8` |

### Tensor Arena Sizing:
- **Peak Consecutive Activation Memory:** 26,880 Bytes (Buffer A + Buffer B double buffering)
- **Runtime Metadata & Alignment Overhead:** ~2,048 Bytes
- **Analytical Minimum Arena Requirement:** 28,928 Bytes (28.25 KB)
- **Configured Tensor Arena in Firmware:** **30,720 Bytes (30.0 KB)**
- **Arena Safety Margin:** **+1,792 Bytes (+5.8%)**

---

## 4. Total Microcontroller SRAM Budget (512 KB Internal SRAM)

| Subsystem / Allocation | Size (Bytes) | Size (KB) | SRAM Occupancy |
| :--- | :---: | :---: | :---: |
| **Tensor Arena (Working Activations)** | 30,720 B | 30.0 KB | 5.86% |
| **FreeRTOS Main Task Stack** | 16,384 B | 16.0 KB | 3.13% |
| **I2S Audio DMA Buffers (Ping-Pong)** | 2,048 B | 2.0 KB | 0.39% |
| **MFCC DSP Scratchpad (FFT/Mel)** | 4,096 B | 4.0 KB | 0.78% |
| **Estimated Total RAM Used** | **53,248 B** | **52.0 KB** | **10.16%** |
| **Available SRAM Headroom** | **460.0 KB** | — | **89.84%** |

---

## 5. Engineering Assumptions & Limitations

- Tensor Arena calculation assumes TFLM single-core runtime with double-buffered greedy activation reuse.
- Flash occupancy is based on .rodata FlatBuffer size (22.3 KB); does not include compiled TFLM kernel library code (~80-120 KB depending on GCC -Os vs -O3).
- RAM estimate does NOT include Wi-Fi or Bluetooth controller heap (which typically consumes 40-80 KB if active).
- Hardware inference speed (~6-10 ms) is an analytical projection based on ESP32-S3 Xtensa PIE vector benchmarks; physical execution latency has not been measured.
- Power consumption (active vs light sleep) is completely unknown without a physical shunt resistor / current probe measurement.
