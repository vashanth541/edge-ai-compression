import hashlib
import os
import re
import numpy as np
import tensorflow as tf

int8_path = 'models/quantized/baseline_dscnn_int8.tflite'
fp32_path = 'models/quantized/baseline_dscnn_fp32.tflite'
header_path = 'deployment/esp32/include/model_data.h'

print('=== FILE SIZES ===')
int8_size = os.path.getsize(int8_path)
fp32_size = os.path.getsize(fp32_path)
header_size = os.path.getsize(header_path)
print(f'INT8 .tflite size: {int8_size} bytes ({int8_size/1024:.2f} KB)')
print(f'FP32 .tflite size: {fp32_size} bytes ({fp32_size/1024:.2f} KB)')
print(f'C Header size on disk: {header_size} bytes ({header_size/1024:.2f} KB)')

with open(int8_path, 'rb') as f:
    int8_bytes = f.read()
int8_sha256 = hashlib.sha256(int8_bytes).hexdigest()
print(f'INT8 .tflite SHA-256: {int8_sha256}')

# Verify C header bytes against .tflite bytes
with open(header_path, 'r', encoding='utf-8') as f:
    header_text = f.read()

# Extract hex bytes
hex_matches = re.findall(r'0x([0-9a-fA-F]{2})', header_text)
header_bytes = bytes([int(h, 16) for h in hex_matches])
header_sha256 = hashlib.sha256(header_bytes).hexdigest()

print(f'Header parsed hex bytes count: {len(header_bytes)}')
print(f'Header SHA-256: {header_sha256}')
print(f'Exact byte match between .tflite and .h: {int8_bytes == header_bytes}')

# Inspect TFLite interpreter
interp = tf.lite.Interpreter(model_path=int8_path)
interp.allocate_tensors()

in_details = interp.get_input_details()
out_details = interp.get_output_details()
tensor_details = interp.get_tensor_details()

print('\n=== INPUT DETAILS ===')
for d in in_details:
    print(f"Name: {d['name']}, Shape: {d['shape']}, Dtype: {d['dtype']}, Quant: {d['quantization']}")

print('\n=== OUTPUT DETAILS ===')
for d in out_details:
    print(f"Name: {d['name']}, Shape: {d['shape']}, Dtype: {d['dtype']}, Quant: {d['quantization']}")

print(f'\nTotal tensors in model: {len(tensor_details)}')
print('\n=== TENSOR TYPES SUMMARY ===')
dtypes_count = {}
for t in tensor_details:
    dt = str(t['dtype'])
    dtypes_count[dt] = dtypes_count.get(dt, 0) + 1
for dt, count in dtypes_count.items():
    print(f'  {dt}: {count} tensors')

print('\n=== ALL TENSORS IN INT8 MODEL ===')
for t in tensor_details:
    q_scale, q_zp = t['quantization']
    print(f"Index {t['index']:2d} | Name: {t['name']:40s} | Shape: {str(list(t['shape'])):16s} | Dtype: {str(t['dtype']):25s} | Scale: {q_scale} | ZP: {q_zp}")
