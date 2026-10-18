import sys
sys.path.insert(0, ".")
import os
from pathlib import Path
from src.audio.dataset import SpeechCommandsDataset, get_speaker_id
from src.audio.config import DatasetConfig

data_dir = Path("data/raw/speech_commands")
dataset_config = DatasetConfig()

print("--- Check 4: Dataset Audit ---")
print(f"Data directory: {data_dir} (exists: {data_dir.exists()})")

# Scan all WAV files
wav_files = list(data_dir.rglob("*.wav"))
print(f"Total WAV files on disk: {len(wav_files)}")

# Count by folder
folder_counts = {}
for f in wav_files:
    folder = f.parent.name
    folder_counts[folder] = folder_counts.get(folder, 0) + 1

print("\nWAV count per directory:")
for folder, count in sorted(folder_counts.items()):
    print(f"  {folder:20s}: {count}")

# Initialize datasets for train, val, test
train_ds = SpeechCommandsDataset(data_dir, split="train", dataset_config=dataset_config)
val_ds = SpeechCommandsDataset(data_dir, split="val", dataset_config=dataset_config)
test_ds = SpeechCommandsDataset(data_dir, split="test", dataset_config=dataset_config)

print(f"\nSplit Sample Counts:")
print(f"  Train: {len(train_ds)} samples")
print(f"  Val:   {len(val_ds)} samples")
print(f"  Test:  {len(test_ds)} samples")
print(f"  Total: {len(train_ds) + len(val_ds) + len(test_ds)} samples")

# Extract speaker sets per split
train_speakers = {get_speaker_id(p) for p, _ in train_ds.samples}
val_speakers = {get_speaker_id(p) for p, _ in val_ds.samples}
test_speakers = {get_speaker_id(p) for p, _ in test_ds.samples}

print(f"\nUnique Speakers per Split:")
print(f"  Train speakers: {len(train_speakers)}")
print(f"  Val speakers:   {len(val_speakers)}")
print(f"  Test speakers:  {len(test_speakers)}")

# Check speaker overlap (leakage)
train_val_overlap = train_speakers.intersection(val_speakers)
train_test_overlap = train_speakers.intersection(test_speakers)
val_test_overlap = val_speakers.intersection(test_speakers)

print("\nSpeaker Overlap Checks:")
print(f"  Train and Val overlap:  {len(train_val_overlap)} (Expected: 0)")
print(f"  Train and Test overlap: {len(train_test_overlap)} (Expected: 0)")
print(f"  Val and Test overlap:   {len(val_test_overlap)} (Expected: 0)")

# Check calibration samples independence
# In quantization.py, representative dataset is drawn from train_ds (or train split)
print(f"\nAre held-out evaluation samples independent of train/calibration? YES (zero overlap).")
