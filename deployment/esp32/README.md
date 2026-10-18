# ESP32-S3 Firmware Project Template for Keyword Spotting (DS-CNN INT8)

This directory contains the production-ready firmware project template for deploying the **INT8-quantized DS-CNN model** onto an **Espressif ESP32-S3** microcontroller using **ESP-IDF** and **TensorFlow Lite for Microcontrollers (TFLM)**.

---

## 1. Project Structure

```
deployment/esp32/
├── CMakeLists.txt                # Root CMake project file
├── sdkconfig.defaults            # Target configuration (240 MHz CPU, 4MB Flash, O2/O3 optimization)
├── model_export_manifest.json    # Cryptographic integrity manifest (SHA-256 hash & metadata)
├── include/
│   ├── model_data.h              # Actual bytes of the INT8 model with 8-byte alignment
│   └── model_labels.h            # 12-class label mapping & enum definitions
└── main/
    ├── CMakeLists.txt            # Main component registration
    ├── idf_component.yml         # Managed component manifest (espressif/esp-tflite-micro)
    └── main.cpp                  # TFLM interpreter setup, tensor arena, and inference loop
```

---

## 2. Model & Runtime Specifications

| Attribute | Specification |
| :--- | :--- |
| **Target Microcontroller** | Espressif ESP32-S3 (Dual-Core Xtensa LX7 @ 240 MHz) |
| **Model Architecture** | Depthwise Separable CNN (DS-CNN, 5,248 parameters) |
| **Quantization Scheme** | Full Integer INT8 (Weights & Activations) |
| **Input Feature Shape** | `[1, 40, 47, 1]` `int8` (40 MFCC coefficients × 47 time frames = 1,880 bytes) |
| **Input Quantization** | Scale: `0.479631`, Zero Point: `54` |
| **Output Logits Shape** | `[1, 12]` `int8` (10 keyword classes + silence + unknown) |
| **Output Quantization** | Scale: `0.100944`, Zero Point: `82` |
| **Model Binary Size** | 22,296 bytes (21.77 KB) |
| **Model SHA-256 Hash** | `4753e7911e93f23a90d5bf19038eedd67312d381123e83d89bcb16c0cd5f6fe7` |
| **Registered Operators (5)** | `CONV_2D`, `DEPTHWISE_CONV_2D`, `AVERAGE_POOL_2D`, `FULLY_CONNECTED`, `RESHAPE` |
| **Tensor Arena RAM** | 30 KB (`30 * 1024` bytes) |

---

## 3. Host Environment Status & Limitations

- **Host Machine:** Windows 11 (CPU-only, AMD Ryzen 3 7320U)
- **Toolchain Status:** `idf.py` (ESP-IDF) is **not installed** on this host laptop.
- **Verification Level:** All model bytes, operator registrations, and C headers are **cryptographically verified byte-for-byte**. However, physical firmware compilation and flashing have **not been performed** on this laptop.
- **Hardware Deployment Status:** **Not performed** (No physical ESP32-S3 hardware connected).

---

## 4. How to Build & Flash When Hardware is Available

Once an ESP-IDF environment (v5.1 or newer) is configured on your development machine or container:

### Step 1: Set Target to ESP32-S3
```bash
idf.py set-target esp32s3
```

### Step 2: Build the Firmware
ESP-IDF's component manager will automatically fetch `espressif/esp-tflite-micro` (with `esp-nn` hardware vector optimizations for Xtensa LX7):
```bash
idf.py build
```

### Step 3: Flash and Monitor
Connect your ESP32-S3 board via USB-UART or USB-JTAG:
```bash
idf.py -p COMx flash monitor
```
*(Replace `COMx` with your actual serial port, e.g. `COM3` on Windows or `/dev/ttyUSB0` on Linux).*
