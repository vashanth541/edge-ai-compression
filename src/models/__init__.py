"""
Neural network architectures package for Keyword Spotting.
"""

from src.models.dscnn import DSCNN, DSCNNConfig, DepthwiseSeparableBlock

__all__ = ["DSCNN", "DSCNNConfig", "DepthwiseSeparableBlock"]
