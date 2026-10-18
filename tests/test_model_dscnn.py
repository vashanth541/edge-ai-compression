"""
Unit tests for the baseline DS-CNN model.
"""

import unittest
import torch

from src.models.dscnn import DSCNN, DSCNNConfig


class TestDSCNN(unittest.TestCase):
    def setUp(self):
        self.config = DSCNNConfig(
            in_channels=1,
            in_features=40,
            in_time_frames=47,
            num_classes=12,
            first_conv_channels=28,
            ds_blocks=4,
        )
        self.model = DSCNN(self.config)

    def test_model_instantiation_and_parameter_count(self):
        """Verify model instantiates and has a compact edge-friendly parameter count."""
        total_params, trainable_params = self.model.count_parameters()
        self.assertEqual(total_params, trainable_params)
        # Should be compact for ESP32 (< 50,000 parameters)
        self.assertLess(total_params, 50000)
        self.assertGreater(total_params, 3000)

    def test_forward_pass_output_shape_and_finite(self):
        """Verify forward pass with batch size 4 produces [4, 12] finite logits."""
        batch_size = 4
        x = torch.randn(batch_size, 1, 40, 47)
        logits = self.model(x)

        self.assertEqual(logits.shape, (batch_size, 12))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(logits).all(), "Logits contain NaN or Inf values")

    def test_invalid_input_dimensions_raise_error(self):
        """Verify ValueError is raised when input is not a 4D tensor."""
        x_3d = torch.randn(4, 40, 47)
        with self.assertRaises(ValueError):
            self.model(x_3d)

    def test_backward_pass_gradients(self):
        """Verify backward pass computes valid gradients without NaN."""
        self.model.train()
        x = torch.randn(2, 1, 40, 47)
        target = torch.tensor([1, 5], dtype=torch.long)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        optimizer.zero_grad()
        logits = self.model(x)
        loss = criterion(logits, target)
        loss.backward()

        self.assertTrue(torch.isfinite(loss).all())
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing for {name}")
                self.assertTrue(torch.isfinite(param.grad).all(), f"NaN gradient in {name}")


if __name__ == "__main__":
    unittest.main()
