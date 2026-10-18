"""
Unit tests for INT8 TFLite model export, quantization parity, and interpreter execution.
"""

from pathlib import Path
import unittest

import numpy as np
import tensorflow as tf
import torch

from src.models.dscnn import DSCNN, DSCNNConfig
from src.quantization.export_int8 import build_keras_dscnn, transfer_weights_pytorch_to_keras, verify_numerical_parity


class TestINT8Quantization(unittest.TestCase):
    def setUp(self):
        self.quantized_dir = Path("models/quantized")
        self.int8_model_path = self.quantized_dir / "baseline_dscnn_int8.tflite"
        self.fp32_model_path = self.quantized_dir / "baseline_dscnn_fp32.tflite"
        self.checkpoint_path = Path("models/checkpoints/baseline_dscnn.pt")

    def test_exported_model_files_exist_and_size_bounds(self):
        """Verify exported TFLite files exist and meet edge micro size constraints (< 35 KB)."""
        self.assertTrue(self.int8_model_path.exists(), f"INT8 model missing at {self.int8_model_path}")
        int8_size = self.int8_model_path.stat().st_size
        self.assertLess(int8_size, 35 * 1024, f"INT8 model exceeds 35 KB ({int8_size / 1024:.2f} KB)")
        self.assertGreater(int8_size, 10 * 1024, f"INT8 model unreasonably small ({int8_size / 1024:.2f} KB)")

        if self.fp32_model_path.exists():
            fp32_size = self.fp32_model_path.stat().st_size
            self.assertGreater(fp32_size, int8_size, "FP32 model should be larger than INT8 model")

    def test_int8_tflite_tensor_types_and_quantization_parameters(self):
        """Verify input and output tensors are strictly int8 with valid scale and zero-point parameters."""
        interpreter = tf.lite.Interpreter(model_path=str(self.int8_model_path))
        interpreter.allocate_tensors()

        in_details = interpreter.get_input_details()[0]
        out_details = interpreter.get_output_details()[0]

        self.assertEqual(in_details["dtype"], np.int8, "Input tensor must be int8")
        self.assertEqual(out_details["dtype"], np.int8, "Output tensor must be int8")

        in_scale, in_zp = in_details["quantization"]
        self.assertGreater(in_scale, 0.0, "Input scale must be positive")
        self.assertIsInstance(in_zp, (int, np.integer))

        out_scale, out_zp = out_details["quantization"]
        self.assertGreater(out_scale, 0.0, "Output scale must be positive")
        self.assertIsInstance(out_zp, (int, np.integer))

        self.assertEqual(list(in_details["shape"]), [1, 40, 47, 1])
        self.assertEqual(list(out_details["shape"]), [1, 12])

    def test_int8_inference_produces_valid_predictions(self):
        """Verify executing INT8 interpreter produces 12-class integer logits within int8 range."""
        interpreter = tf.lite.Interpreter(model_path=str(self.int8_model_path))
        interpreter.allocate_tensors()

        in_idx = interpreter.get_input_details()[0]["index"]
        out_idx = interpreter.get_output_details()[0]["index"]

        dummy_input = np.random.randint(-128, 127, size=(1, 40, 47, 1), dtype=np.int8)
        interpreter.set_tensor(in_idx, dummy_input)
        interpreter.invoke()

        output = interpreter.get_tensor(out_idx)
        self.assertEqual(output.shape, (1, 12))
        self.assertEqual(output.dtype, np.int8)

        pred_class = int(np.argmax(output[0]))
        self.assertTrue(0 <= pred_class < 12)

    def test_pytorch_to_keras_weight_transfer_parity(self):
        """Verify weight transfer function achieves < 1e-4 parity between PyTorch and Keras forward passes."""
        config = DSCNNConfig(
            in_channels=1, in_features=40, in_time_frames=47, num_classes=12, first_conv_channels=28, ds_blocks=4
        )
        pt_model = DSCNN(config)
        keras_model = build_keras_dscnn(config)

        transfer_weights_pytorch_to_keras(pt_model.state_dict(), keras_model, ds_blocks=4)
        max_diff = verify_numerical_parity(pt_model, keras_model, tolerance=1e-4)
        self.assertLess(max_diff, 1e-4, f"Parity difference {max_diff:.4e} exceeds 1e-4")


if __name__ == "__main__":
    unittest.main()
