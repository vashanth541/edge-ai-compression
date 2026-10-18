"""
Configuration definitions for audio processing and dataset handling.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class AudioConfig:
    """Audio signal processing and feature extraction parameters."""

    sample_rate: int = 16000
    clip_duration_ms: int = 1000
    target_samples: int = 16000
    channels: int = 1

    window_size_ms: float = 40.0
    window_stride_ms: float = 20.0
    win_length: int = 640
    hop_length: int = 320
    n_fft: int = 1024

    n_mels: int = 40
    f_min: float = 20.0
    f_max: float = 4000.0

    n_mfcc: int = 40
    log_mels: bool = True

    def __post_init__(self):
        """Validate and synchronize calculated audio properties."""
        calculated_samples = int(self.sample_rate * (self.clip_duration_ms / 1000.0))
        if self.target_samples != calculated_samples:
            self.target_samples = calculated_samples

        calculated_win = int(self.sample_rate * (self.window_size_ms / 1000.0))
        if self.win_length != calculated_win:
            self.win_length = calculated_win

        calculated_hop = int(self.sample_rate * (self.window_stride_ms / 1000.0))
        if self.hop_length != calculated_hop:
            self.hop_length = calculated_hop

        if self.win_length > self.n_fft:
            raise ValueError(
                f"win_length ({self.win_length}) cannot be larger than n_fft ({self.n_fft})"
            )
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.channels != 1:
            raise ValueError("Edge audio pipeline requires mono (channels=1)")

    @property
    def expected_time_frames(self) -> int:
        """Calculate expected number of time frames for a 1-second clip (center=False)."""
        # In PyTorch STFT with center=False, frames slide with step hop_length across (target_samples - n_fft)
        return (self.target_samples - self.n_fft) // self.hop_length + 1


@dataclass
class DatasetConfig:
    """Speech Commands dataset split and labeling parameters."""

    target_commands: List[str] = field(
        default_factory=lambda: [
            "yes",
            "no",
            "up",
            "down",
            "left",
            "right",
            "on",
            "off",
            "stop",
            "go",
        ]
    )
    silence_label: str = "_silence_"
    unknown_label: str = "_unknown_"
    num_classes: int = 12

    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    random_seed: int = 42

    def get_class_to_idx(self) -> Dict[str, int]:
        """
        Generate stable, deterministic label-to-index mapping.
        
        Ordering policy:
        Indices 0 to 9: Alphabetically sorted 10 target commands
        Index 10: silence_label ('_silence_')
        Index 11: unknown_label ('_unknown_')
        """
        sorted_targets = sorted(self.target_commands)
        mapping = {cmd: idx for idx, cmd in enumerate(sorted_targets)}
        mapping[self.silence_label] = len(sorted_targets)
        mapping[self.unknown_label] = len(sorted_targets) + 1
        return mapping

    def get_idx_to_class(self) -> Dict[int, str]:
        """Generate reverse index-to-label mapping."""
        class_to_idx = self.get_class_to_idx()
        return {idx: label for label, idx in class_to_idx.items()}


def load_config(config_path: Optional[str] = None) -> tuple[AudioConfig, DatasetConfig]:
    """
    Load configuration from YAML file or return defaults.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Tuple of (AudioConfig, DatasetConfig).
    """
    if config_path is None:
        default_path = Path(__file__).resolve().parent.parent.parent / "configs" / "audio_config.yaml"
        if default_path.exists():
            config_path = str(default_path)

    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        
        audio_kwargs = raw.get("audio", {})
        dataset_kwargs = raw.get("dataset", {})
        return AudioConfig(**audio_kwargs), DatasetConfig(**dataset_kwargs)

    return AudioConfig(), DatasetConfig()
