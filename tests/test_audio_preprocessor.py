"""
Unit tests for AudioPreprocessor and audio feature extraction.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from src.audio.config import AudioConfig
from src.audio.preprocessor import AudioPreprocessor


class TestAudioPreprocessor(unittest.TestCase):
    def setUp(self):
        self.config = AudioConfig(
            sample_rate=16000,
            clip_duration_ms=1000,
            window_size_ms=40.0,
            window_stride_ms=20.0,
            n_mels=40,
            n_mfcc=40,
        )
        self.preprocessor = AudioPreprocessor(self.config)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_temp_wav(self, duration_s: float = 1.0, sr: int = 16000, channels: int = 1) -> Path:
        """Helper to create a temporary WAV file with synthetic sine wave."""
        samples = int(duration_s * sr)
        t = np.linspace(0, duration_s, samples, endpoint=False)
        data = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

        if channels > 1:
            data = np.stack([data] * channels, axis=-1)

        path = Path(self.temp_dir.name) / f"temp_{duration_s}_{sr}_{channels}.wav"
        sf.write(str(path), data, sr)
        return path

    def test_load_valid_wav(self):
        """Verify successful WAV loading, sample rate, and tensor conversion."""
        path = self._create_temp_wav(duration_s=1.0, sr=16000, channels=1)
        waveform, sr = self.preprocessor.load_wav(path)
        self.assertEqual(sr, 16000)
        self.assertEqual(waveform.shape, (1, 16000))
        self.assertEqual(waveform.dtype, torch.float32)

    def test_load_missing_file_raises_error(self):
        """Verify FileNotFoundError is raised when file does not exist."""
        with self.assertRaises(FileNotFoundError):
            self.preprocessor.load_wav("non_existent_file.wav")

    def test_load_corrupted_file_raises_error(self):
        """Verify ValueError is raised for corrupted audio files."""
        corrupt_path = Path(self.temp_dir.name) / "corrupt.wav"
        corrupt_path.write_text("NOT_A_VALID_WAV_HEADER", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.preprocessor.load_wav(corrupt_path)

    def test_to_mono_conversion(self):
        """Verify multi-channel audio is converted to single-channel mono."""
        # 2-channel stereo
        stereo = torch.randn(2, 16000)
        mono = self.preprocessor.to_mono(stereo)
        self.assertEqual(mono.shape, (1, 16000))

        # Check values are mean of channels
        expected = stereo.mean(dim=0, keepdim=True)
        self.assertTrue(torch.allclose(mono, expected, atol=1e-6))

        # 1-channel pass-through
        single = torch.randn(1, 16000)
        mono_single = self.preprocessor.to_mono(single)
        self.assertEqual(mono_single.shape, (1, 16000))

    def test_resampling(self):
        """Verify resampling from various sample rates to 16 kHz."""
        for orig_sr in [8000, 22050, 44100]:
            orig_len = int(orig_sr * 1.0)
            waveform = torch.randn(1, orig_len)
            resampled = self.preprocessor.resample(waveform, orig_sr, target_sr=16000)
            # Allow +- 2 samples due to fractional resampler filter boundaries
            self.assertAlmostEqual(resampled.shape[-1], 16000, delta=2)

    def test_pad_or_crop_exact(self):
        """Verify that padding and cropping produce exactly 16000 samples."""
        # Exact length: 16000
        w_exact = torch.randn(1, 16000)
        res_exact = self.preprocessor.pad_or_crop(w_exact)
        self.assertEqual(res_exact.shape, (1, 16000))

        # Shorter: 8000 samples (0.5s) -> zero-padded to 16000
        w_short = torch.ones(1, 8000)
        res_short = self.preprocessor.pad_or_crop(w_short)
        self.assertEqual(res_short.shape, (1, 16000))
        # Center padding: middle 8000 samples should be 1.0, boundaries 0.0
        self.assertEqual(res_short[0, 0].item(), 0.0)
        self.assertEqual(res_short[0, -1].item(), 0.0)
        self.assertEqual(res_short[0, 8000].item(), 1.0)

        # Longer: 24000 samples (1.5s) -> center-cropped to 16000
        w_long = torch.randn(1, 24000)
        res_long = self.preprocessor.pad_or_crop(w_long)
        self.assertEqual(res_long.shape, (1, 16000))

    def test_mfcc_feature_shape_and_values(self):
        """Verify MFCC features match expected dimensions (1, 40, 47) and are finite."""
        waveform = torch.randn(1, 16000)
        mfcc = self.preprocessor.extract_mfcc(waveform)

        expected_time_frames = self.config.expected_time_frames
        self.assertEqual(mfcc.shape, (1, 40, expected_time_frames))
        self.assertEqual(mfcc.dtype, torch.float32)
        self.assertTrue(torch.isfinite(mfcc).all(), "MFCC tensor contains NaN or Inf values")

    def test_mel_spectrogram_shape_and_values(self):
        """Verify log-Mel spectrogram dimensions and finite values."""
        waveform = torch.randn(1, 16000)
        mel = self.preprocessor.extract_mel_spectrogram(waveform)

        expected_time_frames = self.config.expected_time_frames
        self.assertEqual(mel.shape, (1, 40, expected_time_frames))
        self.assertEqual(mel.dtype, torch.float32)
        self.assertTrue(torch.isfinite(mel).all(), "Log-Mel tensor contains NaN or Inf values")

    def test_process_file_end_to_end(self):
        """Verify end-to-end WAV file processing to MFCC tensor."""
        path = self._create_temp_wav(duration_s=0.75, sr=22050, channels=2)
        features = self.preprocessor.process_file(path, feature_type="mfcc")
        self.assertEqual(features.shape, (1, 40, self.config.expected_time_frames))
        self.assertEqual(features.dtype, torch.float32)
        self.assertTrue(torch.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
