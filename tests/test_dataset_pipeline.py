"""
Unit tests for SpeechCommandsDataset, split integrity, speaker isolation, and batch loading.
"""

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.audio.config import AudioConfig, DatasetConfig
from src.audio.dataset import (
    SpeechCommandsDataset,
    create_synthetic_dataset,
    determine_split_by_hash,
    get_speaker_id,
)


class TestDatasetPipeline(unittest.TestCase):
    def setUp(self):
        self.audio_cfg = AudioConfig()
        self.dataset_cfg = DatasetConfig()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.synthetic_dir = create_synthetic_dataset(
            self.temp_dir.name,
            num_speakers=15,  # sufficient speakers to populate train, val, and test splits
            clips_per_speaker=2,
            sample_rate=16000,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_class_to_idx_mapping(self):
        """Verify 12-class deterministic label mapping and indices."""
        mapping = self.dataset_cfg.get_class_to_idx()
        self.assertEqual(len(mapping), 12)

        # 10 target commands sorted alphabetically: indices 0..9
        sorted_targets = sorted(self.dataset_cfg.target_commands)
        for idx, cmd in enumerate(sorted_targets):
            self.assertEqual(mapping[cmd], idx, f"Expected {cmd} at index {idx}")

        # Silence is index 10, unknown is index 11
        self.assertEqual(mapping[self.dataset_cfg.silence_label], 10)
        self.assertEqual(mapping[self.dataset_cfg.unknown_label], 11)

    def test_get_speaker_id_parsing(self):
        """Verify speaker ID extraction from standard Speech Commands filenames."""
        # Standard Speech Commands pattern
        p1 = Path("data/raw/yes/0a7c2a8d_nohash_0.wav")
        self.assertEqual(get_speaker_id(p1), "0a7c2a8d")

        p2 = Path("data/raw/no/b83c1e2f_nohash_3.wav")
        self.assertEqual(get_speaker_id(p2), "b83c1e2f")

        # Fallback when pattern is missing
        p3 = Path("data/raw/stop/custom_audio.wav")
        self.assertEqual(get_speaker_id(p3), "custom_audio")

    def test_speaker_hash_determinism(self):
        """Verify that speaker hashing is deterministic across calls."""
        speaker = "0a7c2a8d"
        split_1 = determine_split_by_hash(speaker)
        split_2 = determine_split_by_hash(speaker)
        self.assertEqual(split_1, split_2)

    def test_zero_speaker_leakage_between_splits(self):
        """
        CRITICAL TEST: Verify that train, validation, and test splits
        contain completely disjoint speaker sets (zero data leakage).
        """
        train_ds = SpeechCommandsDataset(self.synthetic_dir, split="train", dataset_config=self.dataset_cfg)
        val_ds = SpeechCommandsDataset(self.synthetic_dir, split="val", dataset_config=self.dataset_cfg)
        test_ds = SpeechCommandsDataset(self.synthetic_dir, split="test", dataset_config=self.dataset_cfg)

        train_speakers = {get_speaker_id(p) for p, _ in train_ds.samples}
        val_speakers = {get_speaker_id(p) for p, _ in val_ds.samples}
        test_speakers = {get_speaker_id(p) for p, _ in test_ds.samples}

        # Check disjoint sets
        self.assertEqual(
            train_speakers & val_speakers,
            set(),
            f"Speaker leakage found between Train and Validation: {train_speakers & val_speakers}",
        )
        self.assertEqual(
            train_speakers & test_speakers,
            set(),
            f"Speaker leakage found between Train and Test: {train_speakers & test_speakers}",
        )
        self.assertEqual(
            val_speakers & test_speakers,
            set(),
            f"Speaker leakage found between Validation and Test: {val_speakers & test_speakers}",
        )

    def test_label_assignment_policy(self):
        """Verify target commands, background noise, and unknown words are labeled correctly."""
        train_ds = SpeechCommandsDataset(self.synthetic_dir, split="train", dataset_config=self.dataset_cfg)
        val_ds = SpeechCommandsDataset(self.synthetic_dir, split="val", dataset_config=self.dataset_cfg)
        test_ds = SpeechCommandsDataset(self.synthetic_dir, split="test", dataset_config=self.dataset_cfg)

        all_samples = train_ds.samples + val_ds.samples + test_ds.samples
        self.assertGreater(len(all_samples), 0)

        target_mapping = self.dataset_cfg.get_class_to_idx()
        yes_idx = target_mapping["yes"]
        no_idx = target_mapping["no"]
        silence_idx = target_mapping["_silence_"]
        unknown_idx = target_mapping["_unknown_"]

        for file_path, label_idx in all_samples:
            subfolder = file_path.parent.name
            if subfolder == "yes":
                self.assertEqual(label_idx, yes_idx)
            elif subfolder == "no":
                self.assertEqual(label_idx, no_idx)
            elif subfolder == "_background_noise_":
                self.assertEqual(label_idx, silence_idx)
            elif subfolder == "bed":
                self.assertEqual(label_idx, unknown_idx)

    def test_dataloader_batching_and_shapes(self):
        """Verify PyTorch DataLoader successfully batches audio features."""
        train_ds = SpeechCommandsDataset(self.synthetic_dir, split="train", dataset_config=self.dataset_cfg)
        self.assertGreater(len(train_ds), 0)

        loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        batch_features, batch_labels, batch_paths = next(iter(loader))

        # Expected batch shape: [batch_size, 1, 40, expected_time_frames]
        expected_frames = self.audio_cfg.expected_time_frames
        self.assertEqual(batch_features.shape, (batch_features.shape[0], 1, 40, expected_frames))
        self.assertEqual(batch_features.dtype, torch.float32)
        self.assertTrue(torch.isfinite(batch_features).all())
        self.assertEqual(len(batch_labels), batch_features.shape[0])
        self.assertTrue(all(0 <= lbl < 12 for lbl in batch_labels))


if __name__ == "__main__":
    unittest.main()
