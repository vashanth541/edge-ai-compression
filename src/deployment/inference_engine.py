"""Live inference engine for the Edge AI Compression Reviewer Demonstrator.

Performs live end-to-end audio validation, feature extraction (MFCC), and CPU inference
using both the FP32 PyTorch baseline and the full INT8 TensorFlow Lite model.
"""

from __future__ import annotations

import io
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

from src.audio.config import AudioConfig, DatasetConfig, load_config
from src.audio.preprocessor import AudioPreprocessor
from src.models.dscnn import DSCNN, DSCNNConfig


class AudioValidationError(Exception):
    """Raised when an audio input fails validation criteria."""
    pass


class ModelPredictor:
    """Loads and executes real-time inference on FP32 and INT8 KWS models."""

    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        if project_root is None:
            self.project_root = Path(__file__).resolve().parent.parent.parent
        else:
            self.project_root = Path(project_root).resolve()

        self.audio_cfg, self.dataset_cfg = load_config(
            str(self.project_root / "configs" / "audio_config.yaml")
        )
        self.preprocessor = AudioPreprocessor(self.audio_cfg)
        self.idx_to_class = self.dataset_cfg.get_idx_to_class()
        self.class_names = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]

        # Model Paths
        self.fp32_checkpoint_path = self.project_root / "models" / "checkpoints" / "baseline_dscnn.pt"
        self.int8_tflite_path = self.project_root / "models" / "quantized" / "baseline_dscnn_int8.tflite"

        # Lazy-loaded model instances
        self._pt_model: Optional[DSCNN] = None
        self._tflite_interpreter: Optional[Any] = None
        self._tflite_in_details: Optional[Dict[str, Any]] = None
        self._tflite_out_details: Optional[Dict[str, Any]] = None

    def _load_pytorch_model(self) -> DSCNN:
        if self._pt_model is not None:
            return self._pt_model

        if not self.fp32_checkpoint_path.is_file():
            raise FileNotFoundError(f"PyTorch baseline checkpoint missing: {self.fp32_checkpoint_path}")

        chk = torch.load(self.fp32_checkpoint_path, map_location="cpu")
        model_cfg = DSCNNConfig(**chk.get("model_config", {}))
        model = DSCNN(model_cfg)
        model.load_state_dict(chk["model_state_dict"])
        model.eval()
        self._pt_model = model
        return self._pt_model

    def _load_tflite_model(self) -> Any:
        if self._tflite_interpreter is not None:
            return self._tflite_interpreter

        if not self.int8_tflite_path.is_file():
            raise FileNotFoundError(f"INT8 TFLite model missing: {self.int8_tflite_path}")

        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(self.int8_tflite_path))
        interpreter.allocate_tensors()

        self._tflite_in_details = interpreter.get_input_details()[0]
        self._tflite_out_details = interpreter.get_output_details()[0]
        self._tflite_interpreter = interpreter
        return self._tflite_interpreter

    def validate_audio_input(
        self,
        audio_source: Union[str, Path, bytes, io.BytesIO],
        max_size_bytes: int = 10 * 1024 * 1024,
    ) -> Tuple[np.ndarray, int]:
        """Validate an audio file or bytes against sample rate, channels, and length constraints.

        Returns:
            Tuple of (waveform: np.ndarray 1D float32, sample_rate: int).
        """
        raw_bytes: bytes
        if isinstance(audio_source, (str, Path)):
            p = Path(audio_source)
            if not p.is_file():
                raise AudioValidationError(f"Audio file does not exist: {p}")
            if p.stat().st_size > max_size_bytes:
                raise AudioValidationError(f"Audio file exceeds maximum size limit (10 MB).")
            with open(p, "rb") as f:
                raw_bytes = f.read()
        elif isinstance(audio_source, io.BytesIO):
            raw_bytes = audio_source.getvalue()
        elif isinstance(audio_source, bytes):
            raw_bytes = audio_source
        else:
            raise AudioValidationError(f"Unsupported audio source type: {type(audio_source)}")

        if len(raw_bytes) < 44:  # WAV header minimum length
            raise AudioValidationError("Audio data is too small to be a valid WAV file.")

        try:
            with io.BytesIO(raw_bytes) as bio:
                waveform_np, sample_rate = sf.read(bio, dtype="float32")
        except Exception as e:
            raise AudioValidationError(f"Failed to decode audio with soundfile: {e}")

        # Handle channel reduction if stereo
        if waveform_np.ndim > 1:
            if waveform_np.shape[1] > 2:
                raise AudioValidationError(f"Unsupported channel count: {waveform_np.shape[1]}")
            waveform_np = np.mean(waveform_np, axis=1)

        duration_sec = len(waveform_np) / float(sample_rate)
        if duration_sec < 0.1:
            raise AudioValidationError(f"Audio clip is too short ({duration_sec:.2f}s). Minimum is 0.1s.")
        if duration_sec > 10.0:
            raise AudioValidationError(f"Audio clip is too long ({duration_sec:.2f}s). Maximum is 10.0s.")

        return waveform_np, sample_rate

    def predict(
        self,
        audio_source: Union[str, Path, bytes, io.BytesIO],
    ) -> Dict[str, Any]:
        """Run complete preprocessing and inference on both FP32 and INT8 models.

        Returns:
            Structured dictionary with waveform, MFCC features, and model predictions.
        """
        # 1. Validation & Decoding
        waveform_np, sample_rate = self.validate_audio_input(audio_source)
        waveform_tensor = torch.from_numpy(waveform_np).unsqueeze(0)  # [1, T]

        # 2. Resampling & Standardization
        if sample_rate != self.audio_cfg.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.audio_cfg.sample_rate
            )
            waveform_tensor = resampler(waveform_tensor)

        # Pad or crop to exactly 1.0s (16,000 samples)
        waveform_1s = self.preprocessor.pad_or_crop(waveform_tensor)
        final_waveform = waveform_1s.squeeze(0).numpy()

        # 3. MFCC Feature Extraction (40 coefficients x 47 time frames)
        mfcc_features = self.preprocessor.extract_mfcc(waveform_1s)  # shape [1, 40, 47]
        mfcc_grid_np = mfcc_features.squeeze(0).numpy()  # shape [40, 47]

        # 4. FP32 PyTorch Baseline Inference
        pt_model = self._load_pytorch_model()
        t0_pt = time.perf_counter()
        with torch.no_grad():
            pt_input = mfcc_features.unsqueeze(0)  # [1, 1, 40, 47]
            pt_logits = pt_model(pt_input)
            pt_probs = torch.softmax(pt_logits, dim=1).squeeze(0).numpy()
        pt_latency_ms = (time.perf_counter() - t0_pt) * 1000.0

        pt_pred_idx = int(np.argmax(pt_probs))
        pt_pred_label = self.idx_to_class[pt_pred_idx]
        pt_top3_indices = np.argsort(pt_probs)[::-1][:3]
        pt_top3 = [
            {"label": self.idx_to_class[int(i)], "prob": float(pt_probs[i]), "class_id": int(i)}
            for i in pt_top3_indices
        ]

        # 5. INT8 TFLite Model Inference
        interpreter = self._load_tflite_model()
        in_details = self._tflite_in_details
        out_details = self._tflite_out_details

        in_scale, in_zp = in_details["quantization"]
        out_scale, out_zp = out_details["quantization"]
        in_idx = in_details["index"]
        out_idx = out_details["index"]

        # Format input as NHWC [1, 40, 47, 1]
        x_float = mfcc_features.permute(1, 2, 0).unsqueeze(0).numpy().astype(np.float32)
        x_int8 = np.round(x_float / in_scale + in_zp).clip(-128, 127).astype(np.int8)

        t0_int8 = time.perf_counter()
        interpreter.set_tensor(in_idx, x_int8)
        interpreter.invoke()
        out_raw_int8 = interpreter.get_tensor(out_idx)  # [1, 12]
        int8_latency_ms = (time.perf_counter() - t0_int8) * 1000.0

        # Dequantize logits and compute Softmax
        dequantized_logits = (out_raw_int8.astype(np.float32) - out_zp) * out_scale
        shifted_logits = dequantized_logits[0] - np.max(dequantized_logits[0])
        exp_logits = np.exp(shifted_logits)
        int8_probs = exp_logits / np.sum(exp_logits)

        int8_pred_idx = int(np.argmax(int8_probs))
        int8_pred_label = self.idx_to_class[int8_pred_idx]
        int8_top3_indices = np.argsort(int8_probs)[::-1][:3]
        int8_top3 = [
            {"label": self.idx_to_class[int(i)], "prob": float(int8_probs[i]), "class_id": int(i)}
            for i in int8_top3_indices
        ]

        # 6. Comparison & Consistency
        models_agreed = (pt_pred_idx == int8_pred_idx)

        return {
            "waveform": final_waveform,
            "sample_rate": self.audio_cfg.sample_rate,
            "duration_sec": 1.0,
            "mfcc_grid": mfcc_grid_np,
            "fp32_pytorch": {
                "predicted_label": pt_pred_label,
                "predicted_idx": pt_pred_idx,
                "confidence": float(pt_probs[pt_pred_idx]),
                "top3": pt_top3,
                "all_probabilities": pt_probs.tolist(),
                "latency_ms": round(pt_latency_ms, 2),
                "device": "Laptop CPU (Windows)",
            },
            "int8_tflite": {
                "predicted_label": int8_pred_label,
                "predicted_idx": int8_pred_idx,
                "confidence": float(int8_probs[int8_pred_idx]),
                "top3": int8_top3,
                "all_probabilities": int8_probs.tolist(),
                "latency_ms": round(int8_latency_ms, 2),
                "device": "Laptop CPU (Windows)",
                "input_quantization": {"scale": float(in_scale), "zero_point": int(in_zp)},
                "output_quantization": {"scale": float(out_scale), "zero_point": int(out_zp)},
            },
            "models_agreed": models_agreed,
            "class_names": self.class_names,
        }


def validate_audio_input(
    audio_source: Union[str, Path, bytes, io.BytesIO],
    max_size_bytes: int = 10 * 1024 * 1024,
) -> Tuple[np.ndarray, int]:
    """Standalone audio validation helper function."""
    predictor = ModelPredictor()
    return predictor.validate_audio_input(audio_source, max_size_bytes=max_size_bytes)
