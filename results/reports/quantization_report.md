# Stage 5: INT8 Quantization Evaluation Report

- **FP32 Baseline Accuracy:** 55.41%
- **INT8 Quantized Accuracy:** 54.83%
- **Accuracy Impact:** -0.58 percentage points
- **FP32 Macro-F1:** 0.5110 $\to$ **INT8 Macro-F1:** 0.5179 (+0.0069)
- **PyTorch Checkpoint Size:** 102.93 KB
- **INT8 Model Size:** 21.77 KB (**78.8% reduction**, 4.73x compression)
- **Raw Weight Payload:** 20.99 KB (FP32) $\to$ 5.25 KB (INT8) (**4.0x reduction**)
- **Laptop CPU Latency:** 2.29 ms (FP32) $\to$ 0.30 ms (INT8)

## Per-Class Quantized Performance

| Class Name | Test Support | FP32 F1 | INT8 F1 | INT8 Precision | INT8 Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `down` | 40 | 0.000 | 0.113 | 0.231 | 0.075 |
| `go` | 49 | 0.571 | 0.606 | 0.600 | 0.612 |
| `left` | 44 | 0.241 | 0.406 | 0.560 | 0.318 |
| `no` | 43 | 0.530 | 0.562 | 0.436 | 0.791 |
| `off` | 45 | 0.532 | 0.522 | 0.429 | 0.667 |
| `on` | 44 | 0.608 | 0.527 | 0.511 | 0.545 |
| `right` | 32 | 0.644 | 0.682 | 0.547 | 0.906 |
| `stop` | 49 | 0.516 | 0.490 | 0.490 | 0.490 |
| `up` | 43 | 0.278 | 0.167 | 0.294 | 0.116 |
| `yes` | 50 | 0.785 | 0.771 | 0.804 | 0.740 |
| `_silence_` | 42 | 1.000 | 1.000 | 1.000 | 1.000 |
| `_unknown_` | 37 | 0.426 | 0.369 | 0.429 | 0.324 |
