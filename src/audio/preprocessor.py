"""
Audio Preprocessor module for Keyword Spotting on Edge AI / Microcontrollers.

Provides deterministic, reproducible audio ingestion, standardizing:
- Mono channel conversion
- 16 kHz resampling
- Fixed 1.0-second (16000 samples) windowing via center-cropping or padding
- MFCC (40 coefficients) and log-Mel spectrogram feature extraction
- Validation of shapes, dtypes, and finite values (no NaNs / Infs)
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T

from src.audio.config import AudioConfig


class AudioPreprocessor:
    """Preprocesses audio waveforms into spectrograms or MFCC tensors."""

    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialize preprocessor with given or default configuration.

        Args:
            config: AudioConfig instance.
        """
        self.config = config or AudioConfig()

        # Initialize transform instances once for reuse and thread safety
        # center=False matches embedded microcontroller DSP windowing (no synthetic zero-padding at boundaries)
        melkwargs = {
            "n_fft": self.config.n_fft,
            "win_length": self.config.win_length,
            "hop_length": self.config.hop_length,
            "f_min": self.config.f_min,
            "f_max": self.config.f_max,
            "n_mels": self.config.n_mels,
            "center": False,
            "power": 2.0,
        }

        self.mfcc_transform = T.MFCC(
            sample_rate=self.config.sample_rate,
            n_mfcc=self.config.n_mfcc,
            log_mels=self.config.log_mels,
            melkwargs=melkwargs,
        )

        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            **melkwargs,
        )

    def load_wav(self, file_path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
        """
        Load a WAV audio file from disk with error validation.

        Args:
            file_path: Path to the WAV file.

        Returns:
            Tuple of (waveform tensor [channels, samples], original sample rate).

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file is corrupted or unsupported format.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        try:
            # Using soundfile for robust Windows I/O
            data, sr = sf.read(str(path), dtype="float32")
        except Exception as e:
            raise ValueError(f"Failed to read WAV file {path}: {str(e)}") from e

        # Convert numpy array to torch tensor: [samples] or [samples, channels]
        tensor = torch.from_numpy(data)
        if tensor.ndim == 1:
            # Shape: [1, samples]
            waveform = tensor.unsqueeze(0)
        else:
            # Shape was [samples, channels], transpose to [channels, samples]
            waveform = tensor.transpose(0, 1)

        return waveform, sr

    def to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert multi-channel audio to mono by averaging channels.

        Args:
            waveform: Audio tensor of shape [channels, samples] or [samples].

        Returns:
            Mono audio tensor of shape [1, samples].
        """
        if waveform.ndim == 1:
            return waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            return waveform.mean(dim=0, keepdim=True)
        return waveform

    def resample(
        self, waveform: torch.Tensor, orig_sr: int, target_sr: Optional[int] = None
    ) -> torch.Tensor:
        """
        Resample waveform to target sample rate.

        Args:
            waveform: Audio tensor [1, samples].
            orig_sr: Original sample rate in Hz.
            target_sr: Target sample rate in Hz (defaults to config.sample_rate).

        Returns:
            Resampled waveform tensor [1, new_samples].
        """
        target = target_sr or self.config.sample_rate
        if orig_sr == target:
            return waveform

        resampler = T.Resample(orig_freq=orig_sr, new_freq=target)
        return resampler(waveform)

    def pad_or_crop(
        self, waveform: torch.Tensor, target_samples: Optional[int] = None
    ) -> torch.Tensor:
        """
        Pad with zeros or center-crop waveform to exactly target_samples.

        Args:
            waveform: Audio tensor [1, samples].
            target_samples: Exact number of samples required (defaults to config.target_samples).

        Returns:
            Standardized audio tensor [1, target_samples].
        """
        target = target_samples or self.config.target_samples
        curr_len = waveform.shape[-1]

        if curr_len == target:
            return waveform
        elif curr_len > target:
            # Center-crop
            start = (curr_len - target) // 2
            return waveform[..., start : start + target]
        else:
            # Zero-pad (center padding)
            pad_total = target - curr_len
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return torch.nn.functional.pad(waveform, (pad_left, pad_right), mode="constant", value=0.0)

    def extract_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract MFCC features from 1-second mono waveform.

        Args:
            waveform: Normalized audio tensor [1, 16000].

        Returns:
            MFCC feature tensor of shape [1, n_mfcc, time_frames].
        """
        features = self.mfcc_transform(waveform)
        self.validate_features(features)
        return features

    def extract_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract log-Mel spectrogram features from 1-second mono waveform.

        Args:
            waveform: Normalized audio tensor [1, 16000].

        Returns:
            Log-Mel spectrogram tensor of shape [1, n_mels, time_frames].
        """
        mel = self.mel_transform(waveform)
        # Numerical stability epsilon before logarithm
        log_mel = torch.log(torch.clamp(mel, min=1e-6))
        self.validate_features(log_mel)
        return log_mel

    def validate_features(
        self,
        features: torch.Tensor,
        expected_time_frames: Optional[int] = None,
    ) -> None:
        """
        Validate that feature tensor contains finite float32 values and matches expected shape.

        Args:
            features: Tensor to validate.
            expected_time_frames: If specified, validates time frame count.

        Raises:
            ValueError: If features contain NaN, Inf, or incorrect dimensions.
            TypeError: If tensor is not float32.
        """
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(features)}")

        if features.dtype != torch.float32:
            raise TypeError(f"Expected float32 features, got {features.dtype}")

        if not torch.isfinite(features).all():
            nan_count = torch.isnan(features).sum().item()
            inf_count = torch.isinf(features).sum().item()
            raise ValueError(
                f"Feature tensor contains non-finite values (NaNs: {nan_count}, Infs: {inf_count})"
            )

        if expected_time_frames is not None:
            actual_frames = features.shape[-1]
            if actual_frames != expected_time_frames:
                raise ValueError(
                    f"Expected {expected_time_frames} time frames, got {actual_frames}"
                )

    def process_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        feature_type: str = "mfcc",
    ) -> torch.Tensor:
        """
        End-to-end preprocessing from a raw waveform to normalized feature tensor.

        Pipeline:
        1. Convert to mono
        2. Resample to 16 kHz
        3. Pad/crop to exactly 1.0 second (16000 samples)
        4. Extract features (MFCC or log-mel)
        5. Validate output

        Args:
            waveform: Input audio tensor.
            sample_rate: Input audio sample rate in Hz.
            feature_type: Either 'mfcc' or 'mel'.

        Returns:
            Preprocessed feature tensor [1, n_features, time_frames].
        """
        mono = self.to_mono(waveform)
        resampled = self.resample(mono, sample_rate, self.config.sample_rate)
        standardized = self.pad_or_crop(resampled, self.config.target_samples)

        if feature_type == "mfcc":
            features = self.extract_mfcc(standardized)
        elif feature_type == "mel":
            features = self.extract_mel_spectrogram(standardized)
        else:
            raise ValueError(f"Unknown feature_type '{feature_type}'. Choose 'mfcc' or 'mel'.")

        self.validate_features(features, expected_time_frames=self.config.expected_time_frames)
        return features

    def process_file(
        self, file_path: Union[str, Path], feature_type: str = "mfcc"
    ) -> torch.Tensor:
        """
        End-to-end preprocessing from a WAV file path to feature tensor.

        Args:
            file_path: Path to the WAV file.
            feature_type: Either 'mfcc' or 'mel'.

        Returns:
            Preprocessed feature tensor [1, n_features, time_frames].
        """
        waveform, sr = self.load_wav(file_path)
        return self.process_waveform(waveform, sr, feature_type=feature_type)
