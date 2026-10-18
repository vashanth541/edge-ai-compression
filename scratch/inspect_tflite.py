import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path='models/quantized/baseline_dscnn_int8.tflite')
interpreter.allocate_tensors()

details = interpreter.get_tensor_details()
print(f'Total tensors in TFLite: {len(details)}')

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print('\nInput Details:', input_details)
print('\nOutput Details:', output_details)

print('\nWeight / Bias Tensors in TFLite:')
total_weights = 0
for t in details:
    name = t['name']
    shape = list(t['shape'])
    dtype = str(t['dtype'])
    scale, zp = t['quantization']
    if len(shape) > 0 and ('conv' in name.lower() or 'dense' in name.lower() or 'matmul' in name.lower() or 'depthwise' in name.lower() or 'bias' in name.lower() or 'kernel' in name.lower() or 'weight' in name.lower()):
        print(f"  id={t['index']:2d} | shape={str(shape):18s} | dtype={dtype:25s} | scale={str(scale)[:8]} | zp={str(zp)[:4]} | name={name}")
        import numpy as np
        if dtype.startswith("<class 'numpy.int8'>") or dtype.startswith("<class 'numpy.int32'>"):
            total_weights += int(np.prod(shape))

print(f'\nTotal weight/bias elements in TFLite: {total_weights}')
