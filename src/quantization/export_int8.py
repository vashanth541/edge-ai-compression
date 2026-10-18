"""
Full Integer INT8 Post-Training Quantization (PTQ) for Baseline DS-CNN.
Converts trained PyTorch DS-CNN weights to a fully integer INT8 TFLite model
optimized for embedded edge deployment (ESP32 / TensorFlow Lite Micro).
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, Any, List, Tuple

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import tensorflow as tf
import yaml

from src.audio.config import AudioConfig, DatasetConfig
from src.audio.dataset import SpeechCommandsDataset
from src.models.dscnn import DSCNN, DSCNNConfig


def build_keras_dscnn(dscnn_cfg: DSCNNConfig) -> tf.keras.Model:
    """
    Construct a 1:1 equivalent Keras DS-CNN in native NHWC format.
    Uses explicit ZeroPadding2D to guarantee exact mathematical parity with PyTorch padding.
    """
    inputs = tf.keras.Input(shape=(40, 47, 1), name="input")
    c = dscnn_cfg.first_conv_channels

    # Stem 2D Conv
    x = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)), name="stem_pad")(inputs)
    x = tf.keras.layers.Conv2D(
        c, (3, 3), strides=(2, 2), padding="valid", use_bias=False, name="stem_conv"
    )(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.1, name="stem_bn")(x)
    x = tf.keras.layers.ReLU(name="stem_relu")(x)

    # 4 Depthwise Separable Blocks
    for i in range(dscnn_cfg.ds_blocks):
        # Depthwise
        x = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)), name=f"block_{i}_dw_pad")(x)
        x = tf.keras.layers.DepthwiseConv2D(
            (3, 3), strides=(1, 1), padding="valid", depth_multiplier=1, use_bias=False, name=f"block_{i}_dw_conv"
        )(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.1, name=f"block_{i}_dw_bn")(x)
        x = tf.keras.layers.ReLU(name=f"block_{i}_dw_relu")(x)

        # Pointwise
        x = tf.keras.layers.Conv2D(
            c, (1, 1), strides=(1, 1), padding="valid", use_bias=False, name=f"block_{i}_pw_conv"
        )(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.1, name=f"block_{i}_pw_bn")(x)
        x = tf.keras.layers.ReLU(name=f"block_{i}_pw_relu")(x)

    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)

    # Linear classifier
    outputs = tf.keras.layers.Dense(dscnn_cfg.num_classes, use_bias=True, name="fc")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="dscnn_keras")


def transfer_weights_pytorch_to_keras(
    pt_state_dict: Dict[str, torch.Tensor], keras_model: tf.keras.Model, ds_blocks: int = 4
) -> None:
    """Transfer trained weights and batch normalization parameters from PyTorch to Keras."""
    sd = pt_state_dict

    # Stem weights: PyTorch [out_c, in_c, H, W] -> Keras [H, W, in_c, out_c]
    stem_w = sd["stem.0.weight"].permute(2, 3, 1, 0).cpu().numpy()
    keras_model.get_layer("stem_conv").set_weights([stem_w])
    keras_model.get_layer("stem_bn").set_weights([
        sd["stem.1.weight"].cpu().numpy(),       # gamma
        sd["stem.1.bias"].cpu().numpy(),         # beta
        sd["stem.1.running_mean"].cpu().numpy(),
        sd["stem.1.running_var"].cpu().numpy(),
    ])

    # Depthwise Separable Blocks
    for i in range(ds_blocks):
        # Depthwise weights: PyTorch [c, 1, H, W] -> Keras [H, W, c, 1]
        dw_w = sd[f"ds_blocks.{i}.depthwise.weight"].permute(2, 3, 0, 1).cpu().numpy()
        keras_model.get_layer(f"block_{i}_dw_conv").set_weights([dw_w])
        keras_model.get_layer(f"block_{i}_dw_bn").set_weights([
            sd[f"ds_blocks.{i}.bn_dw.weight"].cpu().numpy(),
            sd[f"ds_blocks.{i}.bn_dw.bias"].cpu().numpy(),
            sd[f"ds_blocks.{i}.bn_dw.running_mean"].cpu().numpy(),
            sd[f"ds_blocks.{i}.bn_dw.running_var"].cpu().numpy(),
        ])

        # Pointwise weights: PyTorch [c, c, 1, 1] -> Keras [1, 1, c, c]
        pw_w = sd[f"ds_blocks.{i}.pointwise.weight"].permute(2, 3, 1, 0).cpu().numpy()
        keras_model.get_layer(f"block_{i}_pw_conv").set_weights([pw_w])
        keras_model.get_layer(f"block_{i}_pw_bn").set_weights([
            sd[f"ds_blocks.{i}.bn_pw.weight"].cpu().numpy(),
            sd[f"ds_blocks.{i}.bn_pw.bias"].cpu().numpy(),
            sd[f"ds_blocks.{i}.bn_pw.running_mean"].cpu().numpy(),
            sd[f"ds_blocks.{i}.bn_pw.running_var"].cpu().numpy(),
        ])

    # Classifier Dense head: PyTorch [12, 28] -> Keras [28, 12]
    fc_w = sd["fc.weight"].T.cpu().numpy()
    fc_b = sd["fc.bias"].cpu().numpy()
    keras_model.get_layer("fc").set_weights([fc_w, fc_b])


def verify_numerical_parity(
    pt_model: torch.nn.Module, keras_model: tf.keras.Model, tolerance: float = 1e-4
) -> float:
    """Verify forward-pass output parity between PyTorch and Keras on identical inputs."""
    np.random.seed(42)
    sample_pt = np.random.randn(8, 1, 40, 47).astype(np.float32)

    pt_model.eval()
    with torch.no_grad():
        out_pt = pt_model(torch.from_numpy(sample_pt)).numpy()

    # Transpose [B, 1, 40, 47] -> [B, 40, 47, 1] for Keras NHWC
    sample_tf = np.transpose(sample_pt, (0, 2, 3, 1))
    out_tf = keras_model(sample_tf).numpy()

    max_diff = float(np.max(np.abs(out_pt - out_tf)))
    if max_diff > tolerance:
        raise RuntimeError(
            f"Numerical parity check failed: max absolute difference {max_diff:.6e} > tolerance {tolerance:.6e}"
        )
    return max_diff


def export_models(config_path: str = "configs/quantization_config.yaml") -> Dict[str, Any]:
    """Execute complete export pipeline for FP32 and INT8 TFLite models."""
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    q_cfg = config.get("quantization", {})
    p_cfg = config.get("paths", {})

    chk_path = Path(p_cfg.get("baseline_checkpoint", "models/checkpoints/baseline_dscnn.pt"))
    if not chk_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {chk_path}")

    out_dir = Path(p_cfg.get("output_dir", "models/quantized"))
    out_dir.mkdir(parents=True, exist_ok=True)

    fp32_tflite_path = Path(p_cfg.get("fp32_tflite_path", "models/quantized/baseline_dscnn_fp32.tflite"))
    int8_tflite_path = Path(p_cfg.get("int8_tflite_path", "models/quantized/baseline_dscnn_int8.tflite"))

    print(f"[INFO] Loading PyTorch baseline checkpoint: {chk_path}")
    checkpoint = torch.load(chk_path, map_location="cpu")
    model_cfg_dict = checkpoint.get("model_config", {})
    dscnn_cfg = DSCNNConfig(**model_cfg_dict) if model_cfg_dict else DSCNNConfig()

    pt_model = DSCNN(dscnn_cfg)
    pt_model.load_state_dict(checkpoint["model_state_dict"])
    pt_model.eval()

    print("[INFO] Constructing equivalent Keras model and transferring weights...")
    keras_model = build_keras_dscnn(dscnn_cfg)
    transfer_weights_pytorch_to_keras(checkpoint["model_state_dict"], keras_model, ds_blocks=dscnn_cfg.ds_blocks)

    tol = float(q_cfg.get("tolerance_fp32_parity", 1e-4))
    max_parity_diff = verify_numerical_parity(pt_model, keras_model, tolerance=tol)
    print(f"[SUCCESS] Parity verified! Max absolute difference: {max_parity_diff:.4e} (Tolerance: {tol})")

    # Step 1: Export unquantized FP32 TFLite model as reference
    print(f"[INFO] Exporting FP32 TFLite reference model -> {fp32_tflite_path}")
    converter_fp32 = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    fp32_tflite_bytes = converter_fp32.convert()
    with open(fp32_tflite_path, "wb") as f:
        f.write(fp32_tflite_bytes)
    fp32_size_kb = len(fp32_tflite_bytes) / 1024.0
    print(f"  --> Saved FP32 TFLite model: {fp32_size_kb:.2f} KB ({len(fp32_tflite_bytes):,} bytes)")

    # Step 2: Prepare representative calibration dataset (training split only)
    data_dir = p_cfg.get("data_dir", "data/raw/speech_commands")
    n_calib = q_cfg.get("representative_samples_count", 150)
    print(f"[INFO] Loading {n_calib} representative training samples for INT8 calibration from: {data_dir}")

    train_ds = SpeechCommandsDataset(
        data_dir=data_dir, split="train", audio_config=AudioConfig(), dataset_config=DatasetConfig()
    )

    def representative_dataset_gen():
        count = min(n_calib, len(train_ds))
        for idx in range(count):
            feat, _, _ = train_ds[idx]
            # PyTorch feat shape: [1, 40, 47] -> Keras NHWC shape: [1, 40, 47, 1]
            x = feat.permute(1, 2, 0).unsqueeze(0).numpy().astype(np.float32)
            yield [x]

    # Step 3: Full Integer INT8 Quantization
    print("[INFO] Converting to Full Integer INT8 TFLite model...")
    converter_int8 = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = representative_dataset_gen
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.int8
    converter_int8.inference_output_type = tf.int8

    int8_tflite_bytes = converter_int8.convert()
    with open(int8_tflite_path, "wb") as f:
        f.write(int8_tflite_bytes)
    int8_size_kb = len(int8_tflite_bytes) / 1024.0
    print(f"  --> Saved INT8 TFLite model: {int8_size_kb:.2f} KB ({len(int8_tflite_bytes):,} bytes)")

    # Step 4: Validate quantized tensor metadata & inspect operators
    interpreter = tf.lite.Interpreter(model_path=str(int8_tflite_path))
    interpreter.allocate_tensors()

    in_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]

    in_dtype = str(in_details["dtype"])
    out_dtype = str(out_details["dtype"])

    if "int8" not in in_dtype.lower() or "int8" not in out_dtype.lower():
        raise RuntimeError(
            f"Model is NOT fully INT8 quantized! Input dtype={in_dtype}, Output dtype={out_dtype}"
        )

    # Inspect all tensor types
    tensor_details = interpreter.get_tensor_details()
    quantized_tensors = sum(1 for t in tensor_details if t["dtype"] in (np.int8, np.int32))
    total_tensors = len(tensor_details)

    export_summary = {
        "baseline_checkpoint": str(chk_path),
        "fp32_tflite_path": str(fp32_tflite_path),
        "int8_tflite_path": str(int8_tflite_path),
        "fp32_size_bytes": len(fp32_tflite_bytes),
        "fp32_size_kb": fp32_size_kb,
        "int8_size_bytes": len(int8_tflite_bytes),
        "int8_size_kb": int8_size_kb,
        "size_reduction_ratio": fp32_size_kb / int8_size_kb if int8_size_kb > 0 else 0.0,
        "size_reduction_percent": (1.0 - int8_size_kb / fp32_size_kb) * 100.0,
        "numerical_parity_diff": max_parity_diff,
        "input_tensor": {
            "name": in_details["name"],
            "shape": in_details["shape"].tolist(),
            "dtype": in_dtype,
            "scale": float(in_details["quantization"][0]),
            "zero_point": int(in_details["quantization"][1]),
        },
        "output_tensor": {
            "name": out_details["name"],
            "shape": out_details["shape"].tolist(),
            "dtype": out_dtype,
            "scale": float(out_details["quantization"][0]),
            "zero_point": int(out_details["quantization"][1]),
        },
        "quantized_tensors": quantized_tensors,
        "total_tensors": total_tensors,
        "is_full_integer": True,
    }

    print("\n=== INT8 Quantization Export Summary ===")
    print(f"FP32 Model Size: {fp32_size_kb:.2f} KB | INT8 Model Size: {int8_size_kb:.2f} KB")
    print(f"Compression Ratio: {export_summary['size_reduction_ratio']:.2f}x ({export_summary['size_reduction_percent']:.1f}% reduction)")
    print(f"Input Type: {in_dtype} (scale={export_summary['input_tensor']['scale']:.4f}, zp={export_summary['input_tensor']['zero_point']})")
    print(f"Output Type: {out_dtype} (scale={export_summary['output_tensor']['scale']:.4f}, zp={export_summary['output_tensor']['zero_point']})")
    print("Full integer INT8 quantization successfully achieved.")
    return export_summary


def main():
    parser = argparse.ArgumentParser(description="Export Full Integer INT8 TFLite Model")
    parser.add_argument("--config", type=str, default="configs/quantization_config.yaml")
    args = parser.parse_args()
    export_models(args.config)


if __name__ == "__main__":
    main()
