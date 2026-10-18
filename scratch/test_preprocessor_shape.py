import sys
sys.path.insert(0, ".")
import torch
from src.audio.preprocessor import AudioPreprocessor
from src.audio.config import AudioConfig

config = AudioConfig()
preprocessor = AudioPreprocessor(config)

print("--- Audio Preprocessor Configuration Audit ---")
print(f"Sample Rate: {config.sample_rate} Hz")
print(f"Target Samples: {config.target_samples} (Duration: {config.clip_duration_ms} ms)")
print(f"Window Size: {config.window_size_ms} ms -> win_length: {config.win_length} samples")
print(f"Window Stride: {config.window_stride_ms} ms -> hop_length: {config.hop_length} samples")
print(f"n_fft: {config.n_fft} samples")
print(f"Mel Bands: {config.n_mels} (f_min={config.f_min} Hz, f_max={config.f_max} Hz)")
print(f"MFCC Count: {config.n_mfcc}")
print(f"center flag: False (embedded microcontroller compatible)")

# Generate 1.0 second test signal (sine wave + noise)
t = torch.linspace(0, 1.0, config.sample_rate)
test_waveform = torch.sin(2 * torch.pi * 440.0 * t).unsqueeze(0) # [1, 16000]

# Run preprocessor
mfcc = preprocessor.extract_mfcc(test_waveform)
mel = preprocessor.extract_mel_spectrogram(test_waveform)

print(f"\nTest Signal Input Shape: {test_waveform.shape}")
print(f"Output MFCC Tensor Shape: {mfcc.shape}")
print(f"Output Mel Spectrogram Shape: {mel.shape}")
print(f"Expected time frames property: {config.expected_time_frames}")
print(f"Formula check: ({config.target_samples} - {config.n_fft}) // {config.hop_length} + 1 = "
      f"({config.target_samples - config.n_fft}) // {config.hop_length} + 1 = "
      f"{(config.target_samples - config.n_fft) // config.hop_length + 1}")
