import json
import os
import sys
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np
import torch
import tensorflow as tf

from src.audio.config import AudioConfig, DatasetConfig
from src.audio.dataset import SpeechCommandsDataset
from src.models.dscnn import DSCNN, DSCNNConfig

# Load test dataset
data_dir = 'data/raw/speech_commands'
test_ds = SpeechCommandsDataset(data_dir=data_dir, split='test')
n_test = len(test_ds)
print(f'Test samples count: {n_test}')

test_features = []
test_targets = []
for i in range(n_test):
    feat, target, _ = test_ds[i]
    test_features.append(feat)
    test_targets.append(target)

# 1. Run FP32 model
chk_path = 'models/checkpoints/baseline_dscnn.pt'
chk = torch.load(chk_path, map_location='cpu')
pt_model = DSCNN(DSCNNConfig(**chk.get('model_config', {})))
pt_model.load_state_dict(chk['model_state_dict'])
pt_model.eval()

y_pred_pt = []
with torch.no_grad():
    for feat in test_features:
        out = pt_model(feat.unsqueeze(0))
        y_pred_pt.append(int(out.argmax(dim=1).item()))

y_t = np.array(test_targets)
y_p_pt = np.array(y_pred_pt)
pt_acc = float(np.mean(y_t == y_p_pt))
pt_correct = int(np.sum(y_t == y_p_pt))

# 2. Run INT8 model
int8_path = 'models/quantized/baseline_dscnn_int8.tflite'
interp = tf.lite.Interpreter(model_path=int8_path)
interp.allocate_tensors()
in_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]
in_scale = float(in_det['quantization'][0])
in_zp = int(in_det['quantization'][1])
in_idx = in_det['index']
out_idx = out_det['index']

y_pred_int8 = []
for feat in test_features:
    x_float = feat.permute(1, 2, 0).unsqueeze(0).numpy().astype(np.float32)
    x_int8 = np.round(x_float / in_scale + in_zp).clip(-128, 127).astype(np.int8)
    interp.set_tensor(in_idx, x_int8)
    interp.invoke()
    out = interp.get_tensor(out_idx)
    y_pred_int8.append(int(np.argmax(out[0])))

y_p_int8 = np.array(y_pred_int8)
int8_acc = float(np.mean(y_t == y_p_int8))
int8_correct = int(np.sum(y_t == y_p_int8))

# Macro F1 calculation
def calc_macro_f1(y_true, y_pred, num_classes=12):
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (2 * prec * rec) / max(prec + rec, 1e-12) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return cm, float(np.mean(f1s))

cm_pt, f1_pt = calc_macro_f1(y_t, y_p_pt)
cm_int8, f1_int8 = calc_macro_f1(y_t, y_p_int8)

print(f'FP32 Correct: {pt_correct}/{n_test} = {pt_acc*100:.2f}% | Macro F1: {f1_pt:.4f}')
print(f'INT8 Correct: {int8_correct}/{n_test} = {int8_acc*100:.2f}% | Macro F1: {f1_int8:.4f}')
print(f'Accuracy Delta: {(int8_acc - pt_acc)*100:+.2f} pp')
print(f'Macro F1 Delta: {f1_int8 - f1_pt:+.4f}')

# Compare with stored values in quantization_comparison.json
with open('results/reports/quantization_comparison.json', 'r') as f:
    stored = json.load(f)

stored_pt_acc = stored['models']['fp32_pytorch']['test_accuracy']
stored_pt_f1 = stored['models']['fp32_pytorch']['macro_f1']
stored_int8_acc = stored['models']['int8_tflite']['test_accuracy']
stored_int8_f1 = stored['models']['int8_tflite']['macro_f1']

print(f'Matches stored FP32 Acc? {abs(pt_acc - stored_pt_acc) < 1e-6}')
print(f'Matches stored FP32 F1?  {abs(f1_pt - stored_pt_f1) < 1e-6}')
print(f'Matches stored INT8 Acc? {abs(int8_acc - stored_int8_acc) < 1e-6}')
print(f'Matches stored INT8 F1?  {abs(f1_int8 - stored_int8_f1) < 1e-6}')
