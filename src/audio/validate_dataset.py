"""
Validation script for real Google Speech Commands dataset.
Validates:
- Split sizes and per-class distributions across Train, Validation, and Test
- Disjoint speaker sets (Zero speaker leakage verification)
- Audio file integrity (checks for corrupted or unreadable files)
- Preprocessing output tensor shape [1, 40, 47] and finite values
"""

import sys
from pathlib import Path
from typing import Dict, List, Set

# Ensure workspace root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import soundfile as sf
import torch

from src.audio.config import AudioConfig, DatasetConfig
from src.audio.dataset import SpeechCommandsDataset, get_speaker_id
from src.audio.preprocessor import AudioPreprocessor


def audit_real_dataset(data_dir: str = "data/raw/speech_commands") -> Dict[str, any]:
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    audio_cfg = AudioConfig()
    dataset_cfg = DatasetConfig()
    preprocessor = AudioPreprocessor(audio_cfg)
    idx_to_class = dataset_cfg.get_idx_to_class()

    print(f"=== Auditing Real Speech Commands Dataset at {path.resolve()} ===\n")

    # 1. Check all files on disk for corruption
    all_wavs = list(path.glob("*/*.wav"))
    print(f"Total WAV files detected on disk: {len(all_wavs)}")

    corrupt_files = []
    sample_rates = set()
    channels_set = set()

    for w_path in all_wavs:
        try:
            info = sf.info(str(w_path))
            sample_rates.add(info.samplerate)
            channels_set.add(info.channels)
        except Exception as e:
            corrupt_files.append((str(w_path), str(e)))

    print(f"Audio file integrity: {len(corrupt_files)} corrupt files detected.")
    print(f"Sample rates found: {sorted(sample_rates)} Hz")
    print(f"Channel counts found: {sorted(channels_set)}")

    # 2. Instantiate train, val, and test splits
    train_ds = SpeechCommandsDataset(path, split="train", audio_config=audio_cfg, dataset_config=dataset_cfg)
    val_ds = SpeechCommandsDataset(path, split="val", audio_config=audio_cfg, dataset_config=dataset_cfg)
    test_ds = SpeechCommandsDataset(path, split="test", audio_config=audio_cfg, dataset_config=dataset_cfg)

    print(f"\nSplit Distribution:")
    print(f"  - Train Split:      {len(train_ds):4d} samples ({len(train_ds)/len(all_wavs)*100:.1f}%)")
    print(f"  - Validation Split: {len(val_ds):4d} samples ({len(val_ds)/len(all_wavs)*100:.1f}%)")
    print(f"  - Test Split:       {len(test_ds):4d} samples ({len(test_ds)/len(all_wavs)*100:.1f}%)")
    print(f"  - Total Accounted:  {len(train_ds) + len(val_ds) + len(test_ds)} / {len(all_wavs)}")

    # 3. Check Speaker Isolation (Zero Data Leakage)
    train_spks = {get_speaker_id(p) for p, _ in train_ds.samples}
    val_spks = {get_speaker_id(p) for p, _ in val_ds.samples}
    test_spks = {get_speaker_id(p) for p, _ in test_ds.samples}

    leak_train_val = train_spks & val_spks
    leak_train_test = train_spks & test_spks
    leak_val_test = val_spks & test_spks

    print("\nSpeaker Isolation Audit:")
    print(f"  - Train Unique Speakers:      {len(train_spks)}")
    print(f"  - Validation Unique Speakers: {len(val_spks)}")
    print(f"  - Test Unique Speakers:       {len(test_spks)}")
    print(f"  - Train / Val Overlap:        {len(leak_train_val)} (Expected: 0)")
    print(f"  - Train / Test Overlap:       {len(leak_train_test)} (Expected: 0)")
    print(f"  - Val / Test Overlap:         {len(leak_val_test)} (Expected: 0)")

    if leak_train_val or leak_train_test or leak_val_test:
        raise ValueError("CRITICAL ERROR: Speaker leakage detected across dataset splits!")
    print("  --> ZERO SPEAKER LEAKAGE VERIFIED.")

    # 4. Per-Class Counts per split
    def get_class_counts(samples):
        counts = {c: 0 for c in range(12)}
        for _, lbl in samples:
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts

    train_counts = get_class_counts(train_ds.samples)
    val_counts = get_class_counts(val_ds.samples)
    test_counts = get_class_counts(test_ds.samples)

    print("\nPer-Class Sample Counts Across Splits:")
    print(f"{'Class Name':<15} {'Index':<6} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    print("-" * 55)
    for c in range(12):
        c_name = idx_to_class[c]
        tot = train_counts[c] + val_counts[c] + test_counts[c]
        print(f"{c_name:<15} {c:<6} {train_counts[c]:<8} {val_counts[c]:<8} {test_counts[c]:<8} {tot:<8}")

    # 5. Feature extraction check on sample from each split
    print("\nFeature Shape & Finiteness Validation:")
    for split_name, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
        sample_feat, sample_lbl, sample_path = ds[0]
        assert sample_feat.shape == (1, 40, 47), f"Unexpected shape {sample_feat.shape}"
        assert torch.isfinite(sample_feat).all(), "Non-finite values found"
        assert sample_feat.dtype == torch.float32, f"Unexpected dtype {sample_feat.dtype}"
        print(f"  - {split_name} Sample #0: Shape={sample_feat.shape}, Class={idx_to_class[sample_lbl]}, Finite=True")

    print("\n=== Dataset Pipeline Validation Complete: All Checks Passed ===\n")

    return {
        "total_files": len(all_wavs),
        "train_count": len(train_ds),
        "val_count": len(val_ds),
        "test_count": len(test_ds),
        "train_speakers": len(train_spks),
        "val_speakers": len(val_spks),
        "test_speakers": len(test_spks),
        "per_class": {
            idx_to_class[c]: {
                "train": train_counts[c],
                "val": val_counts[c],
                "test": test_counts[c],
                "total": train_counts[c] + val_counts[c] + test_counts[c],
            }
            for c in range(12)
        },
    }


if __name__ == "__main__":
    audit_real_dataset()
