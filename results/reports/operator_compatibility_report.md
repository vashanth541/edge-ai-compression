# TFLite Micro / ESP32 Operator Compatibility Report

- **Quantized Model:** `baseline_dscnn_int8.tflite` (21.77 KB)
- **Quantization Scheme:** Full Integer INT8 (Weights & Activations)
- **Input Tensor:** shape `[1, 40, 47, 1]`, dtype `int8`, scale `0.479631`, zero_point `54`
- **Output Tensor:** shape `[1, 12]`, dtype `int8`, scale `0.100944`, zero_point `82`

## Operator Inspection & Embedded Kernel Compatibility

| TFLite Operator | Quantized Type | TFLite Micro Support | ESP-NN (ESP32-S3) Vector Acceleration |
| :--- | :--- | :---: | :---: |
| `CONV_2D` | `int8` weights & activations | **Full** | **Accelerated** (Xtensa LX7 SIMD) |
| `DEPTHWISE_CONV_2D` | `int8` weights & activations | **Full** | **Accelerated** (Xtensa LX7 SIMD) |
| `AVERAGE_POOL_2D` | `int8` | **Full** | **Accelerated** |
| `FULLY_CONNECTED` | `int8` weights & activations | **Full** | **Accelerated** (int8 GEMM) |
| `RESHAPE` | integer metadata | **Full** | Direct pointer stride |

### Compatibility Verdict
All 5 detected operators belong to the standard TensorFlow Lite for Microcontrollers (TFLM) reference kernel registry and are directly supported with hardware-accelerated vector assembly kernels in Espressif's `esp-nn` library for ESP32-S3. No custom operators, Flex delegates, or unsupported floating-point fallback layers are present in the model.
