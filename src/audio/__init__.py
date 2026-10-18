"""
Audio feature extraction, preprocessing, and dataset pipeline package.
"""

from src.audio.config import AudioConfig, DatasetConfig, load_config
from src.audio.dataset import (
    SpeechCommandsDataset,
    create_synthetic_dataset,
    determine_split_by_hash,
    get_speaker_id,
)
from src.audio.preprocessor import AudioPreprocessor

__all__ = [
    "AudioConfig",
    "DatasetConfig",
    "load_config",
    "AudioPreprocessor",
    "SpeechCommandsDataset",
    "create_synthetic_dataset",
    "get_speaker_id",
    "determine_split_by_hash",
]
