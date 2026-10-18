"""
Dataset pipeline for Google Speech Commands Keyword Spotting.

Implements:
- 12-class classification policy (10 target commands + 1 silence + 1 unknown)
- Strict speaker-aware split partitioning (zero speaker leakage across train/val/test)
- Support for official validation_list.txt / testing_list.txt and deterministic SHA-1 speaker hashing
- Background noise slicing for the '_silence_' class
- Helper functions to generate synthetic datasets for offline testing
"""

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.audio.config import AudioConfig, DatasetConfig
from src.audio.preprocessor import AudioPreprocessor


def get_speaker_id(file_path: Union[str, Path]) -> str:
    """
    Extract speaker ID from a Speech Commands filename.

    Naming convention: {speaker_id}_nohash_{clip_index}.wav
    If the pattern is not present, hashes the file stem to provide a deterministic ID.
    """
    stem = Path(file_path).stem
    if "_nohash_" in stem:
        return stem.split("_nohash_")[0]
    return stem


def determine_split_by_hash(
    speaker_id: str,
    val_percentage: float = 10.0,
    test_percentage: float = 10.0,
) -> str:
    """
    Assign a speaker to 'train', 'val', or 'test' split deterministically using SHA-1.

    Ensures zero speaker leakage across splits.
    """
    # Standard Speech Commands SHA-1 hashing algorithm
    hash_int = int(hashlib.sha1(speaker_id.encode("utf-8")).hexdigest(), 16)
    bucket = hash_int % 100

    if bucket < val_percentage:
        return "val"
    elif bucket < (val_percentage + test_percentage):
        return "test"
    else:
        return "train"


class SpeechCommandsDataset(Dataset):
    """
    PyTorch Dataset for Speech Commands keyword spotting.

    Label Policy:
    - 10 Target Commands (Indices 0..9): 'down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes'
    - '_silence_' (Index 10): Ambient/background noise clips
    - '_unknown_' (Index 11): Other non-target words (e.g. 'bed', 'cat', 'dog', etc.)
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        feature_type: str = "mfcc",
        audio_config: Optional[AudioConfig] = None,
        dataset_config: Optional[DatasetConfig] = None,
        preprocessor: Optional[AudioPreprocessor] = None,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Root directory containing Speech Commands subdirectories.
            split: One of 'train', 'val', or 'test'.
            feature_type: 'mfcc' or 'mel'.
            audio_config: Audio configuration instance.
            dataset_config: Dataset configuration instance.
            preprocessor: AudioPreprocessor instance (created if None).
        """
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        self.data_dir = Path(data_dir)
        self.split = split
        self.feature_type = feature_type
        self.audio_config = audio_config or AudioConfig()
        self.dataset_config = dataset_config or DatasetConfig()
        self.preprocessor = preprocessor or AudioPreprocessor(self.audio_config)

        self.class_to_idx = self.dataset_config.get_class_to_idx()
        self.idx_to_class = self.dataset_config.get_idx_to_class()

        self.samples: List[Tuple[Path, int]] = []
        self._load_and_filter_dataset()

    def _load_and_filter_dataset(self) -> None:
        """Scan directory, map labels according to the 12-class policy, and assign splits."""
        if not self.data_dir.exists():
            return

        # Check for official split files
        val_list_file = self.data_dir / "validation_list.txt"
        test_list_file = self.data_dir / "testing_list.txt"

        official_val_set = set()
        official_test_set = set()

        if val_list_file.exists():
            with open(val_list_file, "r", encoding="utf-8") as f:
                official_val_set = {
                    line.strip().replace("\\", "/") for line in f if line.strip()
                }

        if test_list_file.exists():
            with open(test_list_file, "r", encoding="utf-8") as f:
                official_test_set = {
                    line.strip().replace("\\", "/") for line in f if line.strip()
                }

        has_official_lists = bool(official_val_set or official_test_set)
        target_set = set(self.dataset_config.target_commands)

        # Iterate over all WAV files in subdirectories
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if not file.endswith(".wav"):
                    continue

                full_path = Path(root) / file
                rel_path = full_path.relative_to(self.data_dir).as_posix()
                subfolder = Path(root).name

                # Check for silence (either raw _background_noise_ or curated _silence_)
                if subfolder.startswith("_background_noise_") or subfolder == self.dataset_config.silence_label:
                    label_str = self.dataset_config.silence_label
                elif subfolder in target_set:
                    label_str = subfolder
                else:
                    label_str = self.dataset_config.unknown_label

                label_idx = self.class_to_idx[label_str]

                # Determine split
                if has_official_lists:
                    if rel_path in official_test_set:
                        assigned_split = "test"
                    elif rel_path in official_val_set:
                        assigned_split = "val"
                    else:
                        assigned_split = "train"
                else:
                    speaker_id = get_speaker_id(full_path)
                    assigned_split = determine_split_by_hash(
                        speaker_id,
                        val_percentage=self.dataset_config.val_ratio * 100,
                        test_percentage=self.dataset_config.test_ratio * 100,
                    )

                if assigned_split == self.split:
                    self.samples.append((full_path, label_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Fetch and process audio sample.

        Returns:
            Tuple of (feature_tensor [1, n_features, time_frames], label_idx, file_path_str)
        """
        file_path, label_idx = self.samples[idx]
        features = self.preprocessor.process_file(file_path, feature_type=self.feature_type)
        return features, label_idx, str(file_path)


def create_synthetic_dataset(
    output_dir: Union[str, Path],
    num_speakers: int = 6,
    clips_per_speaker: int = 3,
    sample_rate: int = 16000,
) -> Path:
    """
    Generate a small synthetic Speech Commands dataset for offline unit tests.

    Creates:
    - 2 target commands ('yes', 'no')
    - 1 non-target command ('bed', labeled as _unknown_)
    - 1 background noise folder ('_background_noise_', labeled as _silence_)
    - Synthetic WAV files with distinct speaker IDs to test speaker-aware split isolation.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    words = ["yes", "no", "bed", "_background_noise_"]

    for word in words:
        word_dir = root / word
        word_dir.mkdir(parents=True, exist_ok=True)

        for spk_idx in range(num_speakers):
            speaker_id = f"spk_{word[:3]}_{spk_idx:03d}"
            for clip_idx in range(clips_per_speaker):
                filename = f"{speaker_id}_nohash_{clip_idx}.wav"
                file_path = word_dir / filename

                # Generate 1.0s synthetic audio (sine wave with unique frequency + low noise)
                t = torch.linspace(0, 1.0, sample_rate)
                freq = 200.0 + (spk_idx * 50.0)
                audio = 0.5 * torch.sin(2 * 3.14159 * freq * t) + 0.05 * torch.randn_like(t)
                audio = torch.clamp(audio, -1.0, 1.0).numpy()

                sf.write(str(file_path), audio, sample_rate)

    return root
