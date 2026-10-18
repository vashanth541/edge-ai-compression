import sys
sys.path.insert(0, ".")
import json
import hashlib
from pathlib import Path

# 1. Artifact files
pt_path = Path("models/checkpoints/baseline_dscnn.pt")
tflite_path = Path("models/quantized/baseline_dscnn_int8.tflite")
header_path = Path("deployment/esp32/include/model_data.h")
quant_json_path = Path("results/reports/quantization_comparison.json")
baseline_json_path = Path("results/reports/baseline_test_metrics.json")

print("--- Check 3 Audit ---")
print(f"FP32 Checkpoint exists: {pt_path.exists()} | Size: {pt_path.stat().st_size} bytes")
print(f"INT8 TFLite exists: {tflite_path.exists()} | Size: {tflite_path.stat().st_size} bytes")
print(f"C Header exists: {header_path.exists()} | Size: {header_path.stat().st_size} bytes")

# 2. TFLite SHA-256
tflite_bytes = tflite_path.read_bytes()
tflite_sha256 = hashlib.sha256(tflite_bytes).hexdigest()
print(f"TFLite SHA256: {tflite_sha256}")

# 3. Read metrics
quant_data = json.loads(quant_json_path.read_text(encoding="utf-8"))
baseline_data = json.loads(baseline_json_path.read_text(encoding="utf-8"))

print("\nQuantization Comparison JSON Summary:")
print("  test_samples:", quant_data.get("evaluation_metadata", {}).get("test_samples"))
print("  int8 accuracy:", quant_data.get("int8_metrics", {}).get("accuracy"))
print("  int8 macro_f1:", quant_data.get("int8_metrics", {}).get("macro_f1"))
print("  int8 latency_ms:", quant_data.get("int8_metrics", {}).get("latency_ms"))
print("  int8 throughput:", quant_data.get("int8_metrics", {}).get("throughput_samples_per_sec"))

print("\nBaseline JSON Summary:")
print("  fp32 accuracy:", baseline_data.get("accuracy"))
print("  fp32 macro_f1:", baseline_data.get("macro_f1"))
print("  fp32 latency_ms:", baseline_data.get("latency_ms"))
print("  fp32 throughput:", baseline_data.get("throughput_samples_per_sec"))

# Recalculate deltas
fp32_size = pt_path.stat().st_size
int8_size = tflite_path.stat().st_size
size_reduction = (fp32_size - int8_size) / fp32_size * 100.0

fp32_acc = baseline_data["accuracy"]
int8_acc = quant_data["int8_metrics"]["accuracy"]
acc_delta_pp = (int8_acc - fp32_acc) * 100.0

fp32_lat = baseline_data["latency_ms"]
int8_lat = quant_data["int8_metrics"]["latency_ms"]
speedup = fp32_lat / int8_lat

print(f"\nRecalculated Reductions:")
print(f"  File size reduction: ({fp32_size} - {int8_size}) / {fp32_size} = {size_reduction:.2f}% (ratio: {fp32_size/int8_size:.2f}x)")
print(f"  Accuracy delta: ({int8_acc*100:.2f}% - {fp32_acc*100:.2f}%) = {acc_delta_pp:+.2f} percentage points")
print(f"  Laptop Latency speedup: {fp32_lat:.2f} ms / {int8_lat:.2f} ms = {speedup:.2f}x")

# Raw weight parameters
raw_fp32_weights = 5248 * 4 # 20,992 bytes
raw_int8_weights = 5248 * 1 # 5,248 bytes
weight_reduction = (raw_fp32_weights - raw_int8_weights) / raw_fp32_weights * 100.0
print(f"  Raw weight reduction: ({raw_fp32_weights} - {raw_int8_weights}) / {raw_fp32_weights} = {weight_reduction:.2f}% (4.00x)")
