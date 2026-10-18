"""
Depthwise Separable Convolutional Neural Network (DS-CNN) for Keyword Spotting.

Designed for edge deployment and microcontroller inference (ESP32 / TFLite Micro):
- Standard 2D Conv front-end
- Repetitive Depthwise-Separable Convolution blocks (Depthwise Conv + Pointwise Conv)
- Global Average Pooling (eliminating bulky Flatten -> Dense layers)
- Single linear classification head producing 12-class logits
- Pure ReLU activations and static tensor dimensions for ESP-NN acceleration compatibility
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class DSCNNConfig:
    """Hyperparameters and architectural configuration for DS-CNN."""

    in_channels: int = 1
    in_features: int = 40          # 40 MFCC coefficients
    in_time_frames: int = 47       # 47 time frames for 1.0s audio @ 16kHz
    num_classes: int = 12          # 10 target commands + silence + unknown
    first_conv_channels: int = 28  # Number of filters in initial 2D convolution
    ds_blocks: int = 4             # Number of Depthwise Separable blocks
    kernel_size: Tuple[int, int] = (3, 3)
    dropout: float = 0.1


class DepthwiseSeparableBlock(nn.Module):
    """
    Depthwise-Separable Convolution Block:
    1. Depthwise Convolution (groups = channels, spatial filtering)
    2. Batch Normalization + ReLU
    3. Pointwise Convolution (1x1 kernel, channel mixing)
    4. Batch Normalization + ReLU
    """

    def __init__(
        self,
        channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (1, 1),
    ):
        super().__init__()

        # Depthwise convolution: filters each input channel independently
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.bn_dw = nn.BatchNorm2d(channels)
        self.relu_dw = nn.ReLU(inplace=True)

        # Pointwise convolution: 1x1 convolution across all channels
        self.pointwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=False,
        )
        self.bn_pw = nn.BatchNorm2d(channels)
        self.relu_pw = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu_dw(self.bn_dw(self.depthwise(x)))
        out = self.relu_pw(self.bn_pw(self.pointwise(out)))
        return out


class DSCNN(nn.Module):
    """
    Compact DS-CNN model for Edge AI Keyword Spotting.

    Input shape:  [batch_size, 1, 40, 47] (Batch, Channels, MFCCs, TimeFrames)
    Output shape: [batch_size, num_classes] (Logits for cross-entropy loss)
    """

    def __init__(self, config: Optional[DSCNNConfig] = None):
        super().__init__()
        self.config = config or DSCNNConfig()

        c = self.config.first_conv_channels

        # Initial 2D Convolution with stride (2, 2) for immediate spatial reduction
        self.stem = nn.Sequential(
            nn.Conv2d(
                self.config.in_channels,
                c,
                kernel_size=self.config.kernel_size,
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

        # Sequence of Depthwise Separable Blocks
        blocks = []
        for _ in range(self.config.ds_blocks):
            blocks.append(
                DepthwiseSeparableBlock(
                    channels=c,
                    kernel_size=self.config.kernel_size,
                    stride=(1, 1),
                    padding=(1, 1),
                )
            )
        self.ds_blocks = nn.Sequential(*blocks)

        # Global Average Pooling reduces spatial dimensions to (1, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout and linear classifier
        self.dropout = nn.Dropout(p=self.config.dropout)
        self.fc = nn.Linear(c, self.config.num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming normal initialization for conv layers, zeros for biases."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, 1, in_features, in_time_frames].

        Returns:
            Logits tensor [batch_size, num_classes].
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input tensor [batch_size, channels, features, time], got {x.ndim}D with shape {x.shape}"
            )

        out = self.stem(x)
        out = self.ds_blocks(out)
        out = self.gap(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits

    def count_parameters(self) -> Tuple[int, int]:
        """
        Count model parameters.

        Returns:
            Tuple of (total_parameters, trainable_parameters).
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
