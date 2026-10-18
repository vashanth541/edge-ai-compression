# Baseline DS-CNN Evaluation Report

- **Data Source:** `data/raw/speech_commands` (Synthetic: False)
- **Test Set Size:** 518 samples
- **Test Accuracy:** 55.41%
- **Macro-F1 Score:** 0.5110
- **Trainable Parameters:** 5,248
- **Checkpoint Size:** 102.93 KB (105,396 bytes)
- **CPU Inference Latency (Mean):** 1.69 ms ± 0.39 ms
- **CPU Throughput:** 590.9 samples/sec

## Per-Class Evaluation Summary

| Class Name | Support | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| `down` | 40 | 0.000 | 0.000 | 0.000 |
| `go` | 49 | 0.619 | 0.531 | 0.571 |
| `left` | 44 | 0.500 | 0.159 | 0.241 |
| `no` | 43 | 0.393 | 0.814 | 0.530 |
| `off` | 45 | 0.453 | 0.644 | 0.532 |
| `on` | 44 | 0.534 | 0.705 | 0.608 |
| `right` | 32 | 0.509 | 0.875 | 0.644 |
| `stop` | 49 | 0.545 | 0.490 | 0.516 |
| `up` | 43 | 0.345 | 0.233 | 0.278 |
| `yes` | 50 | 0.737 | 0.840 | 0.785 |
| `_silence_` | 42 | 1.000 | 1.000 | 1.000 |
| `_unknown_` | 37 | 0.542 | 0.351 | 0.426 |
