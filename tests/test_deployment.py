"""Unit tests for Virtual ESP32-S3 Deployment and Audio Inference Lab (Stage 7).

Tests C-array byte integrity, label mapping consistency, virtual resource calculations,
audio input validation, and real-time CPU model inference.
"""

from __future__ import annotations

import io
from pathlib import Path
import re

import numpy as np
import pytest
import soundfile as sf

from src.dashboard.data_loader import get_project_root
from src.deployment.inference_engine import AudioValidationError, ModelPredictor, validate_audio_input
from src.deployment.resource_analyzer import ResourceAnalyzer, analyze_model_resources
from src.export.export_c_header import verify_c_header_integrity


class TestCHeaderExport:
    """Test suite for C array model export and cryptographic integrity."""

    def test_c_header_byte_integrity(self) -> None:
        """Verify that model_data.h reconstructs the exact byte sequence and SHA-256 of the TFLite model."""
        root = get_project_root()
        tflite_path = root / "models" / "quantized" / "baseline_dscnn_int8.tflite"
        c_header_path = root / "deployment" / "esp32" / "include" / "model_data.h"

        assert tflite_path.is_file(), f"Source TFLite model missing: {tflite_path}"
        assert c_header_path.is_file(), f"Exported C header missing: {c_header_path}"

        is_valid, details = verify_c_header_integrity(c_header_path, tflite_path)

        assert is_valid is True, f"C header bytes do not match source TFLite model! Details: {details}"
        assert details["length_match"] is True
        assert details["sha256_match"] is True
        assert details["exact_byte_match"] is True
        assert details["original_length"] == tflite_path.stat().st_size

    def test_labels_header_consistency(self) -> None:
        """Verify that model_labels.h defines all 12 classes with proper enum indices."""
        root = get_project_root()
        labels_header_path = root / "deployment" / "esp32" / "include" / "model_labels.h"

        assert labels_header_path.is_file(), f"Labels header missing: {labels_header_path}"
        content = labels_header_path.read_text(encoding="utf-8")

        expected_classes = [
            "down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes",
            "_silence_", "_unknown_",
        ]
        assert "MODEL_NUM_CLASSES 12" in content
        for cls_name in expected_classes:
            assert f'"{cls_name}"' in content, f"Class '{cls_name}' not found in labels header"

    def test_manifest_file_referenced_in_readme(self) -> None:
        """Verify the export manifest is referenced in the firmware README."""
        root = get_project_root()
        readme_path = root / "deployment" / "esp32" / "README.md"
        assert readme_path.is_file()
        content = readme_path.read_text(encoding="utf-8")
        assert "model_export_manifest.json" in content


class TestFirmwareTemplate:
    """Test suite for the ESP32-S3 firmware project structure."""

    def test_firmware_files_exist(self) -> None:
        """Verify all essential firmware project files are present."""
        root = get_project_root()
        esp32_dir = root / "deployment" / "esp32"

        required_files = [
            esp32_dir / "CMakeLists.txt",
            esp32_dir / "sdkconfig.defaults",
            esp32_dir / "main" / "CMakeLists.txt",
            esp32_dir / "main" / "idf_component.yml",
            esp32_dir / "main" / "main.cpp",
            esp32_dir / "include" / "model_data.h",
            esp32_dir / "include" / "model_labels.h",
            esp32_dir / "README.md",
        ]

        for req in required_files:
            assert req.is_file(), f"Required firmware file missing: {req}"

    def test_firmware_main_registers_operators(self) -> None:
        """Verify main.cpp registers the required 5 TFLM operators."""
        root = get_project_root()
        main_cpp = (root / "deployment" / "esp32" / "main" / "main.cpp").read_text(encoding="utf-8")

        required_kernels = [
            "AddConv2D",
            "AddDepthwiseConv2D",
            "AddAveragePool2D",
            "AddFullyConnected",
            "AddReshape",
        ]
        for kernel in required_kernels:
            assert kernel in main_cpp, f"Operator {kernel} not registered in main.cpp"

    def test_firmware_readme_has_toolchain_disclosure(self) -> None:
        """Verify README clearly states host toolchain limitations and software-only status."""
        root = get_project_root()
        readme = (root / "deployment" / "esp32" / "README.md").read_text(encoding="utf-8")

        assert "idf.py" in readme, "README must mention idf.py toolchain"
        assert "not installed" in readme.lower() or "not performed" in readme.lower(), \
            "README must disclose that toolchain is not installed or deployment not performed"


class TestResourceAnalyzer:
    """Test suite for analytical embedded resource calculations."""

    def test_resource_analysis_structure(self) -> None:
        """Verify that resource analyzer returns all required sections and valid values."""
        root = get_project_root()
        res = analyze_model_resources(root)

        assert "model_storage" in res
        assert "flash_partition" in res
        assert "tensor_arena" in res
        assert "sram_ram_budget" in res
        assert "assumptions_and_limitations" in res

        # Check values
        assert res["model_storage"]["tflite_bytes"] == 22296
        assert res["model_storage"]["total_parameters"] == 5248
        assert res["flash_partition"]["total_flash_occupancy_pct"] < 1.0  # < 1% of 4MB Flash

    def test_tensor_arena_safety_margin(self) -> None:
        """Verify configured tensor arena provides a positive safety margin over analytical minimum."""
        root = get_project_root()
        res = analyze_model_resources(root)

        ta = res["tensor_arena"]
        assert ta["configured_arena_bytes"] > ta["analytical_min_arena_bytes"]
        assert ta["safety_margin_pct"] >= 5.0  # At least 5% safety margin

    def test_sram_headroom_feasibility(self) -> None:
        """Verify total SRAM usage leaves abundant headroom (>75%) on ESP32-S3 (512 KB)."""
        root = get_project_root()
        res = analyze_model_resources(root)

        sb = res["sram_ram_budget"]
        assert sb["sram_occupancy_pct"] < 25.0  # Less than 25% of 512 KB
        assert sb["sram_headroom_pct"] > 75.0  # Greater than 75% free


class TestAudioInferenceEngine:
    """Test suite for audio validation and live inference."""

    def test_validate_audio_valid_sample(self) -> None:
        """Verify validation on genuine bundled sample WAV file."""
        root = get_project_root()
        sample_path = root / "data" / "sample_audio" / "yes.wav"
        assert sample_path.is_file(), f"Sample audio missing: {sample_path}"

        predictor = ModelPredictor(project_root=root)
        waveform, sr = predictor.validate_audio_input(sample_path)
        assert isinstance(waveform, np.ndarray)
        assert sr == 16000
        assert len(waveform) == 16000

    def test_validate_audio_too_short(self) -> None:
        """Verify validation rejects audio shorter than 0.1s."""
        # 0.05 seconds of audio at 16000 Hz = 800 samples
        short_wave = np.zeros(800, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, short_wave, 16000, format="WAV")
        buf.seek(0)

        predictor = ModelPredictor()
        with pytest.raises(AudioValidationError, match="too short"):
            predictor.validate_audio_input(buf)

    def test_validate_audio_corrupted_bytes(self) -> None:
        """Verify validation rejects corrupted bytes."""
        corrupted = b"RIFF1234WAVEfmt " + b"\x00" * 10
        predictor = ModelPredictor()
        with pytest.raises(AudioValidationError):
            predictor.validate_audio_input(corrupted)

    def test_predictor_live_prediction(self) -> None:
        """Verify ModelPredictor runs end-to-end CPU inference on both FP32 and INT8 models."""
        root = get_project_root()
        sample_path = root / "data" / "sample_audio" / "yes.wav"
        predictor = ModelPredictor(project_root=root)

        result = predictor.predict(sample_path)

        assert "fp32_pytorch" in result
        assert "int8_tflite" in result
        assert "models_agreed" in result
        assert "class_names" in result
        assert len(result["class_names"]) == 12

        # FP32 checks
        pt = result["fp32_pytorch"]
        assert pt["predicted_label"] in result["class_names"]
        assert 0.0 <= pt["confidence"] <= 1.0
        assert pt["latency_ms"] > 0.0
        assert len(pt["top3"]) == 3
        assert pytest.approx(sum(pt["all_probabilities"]), abs=1e-3) == 1.0

        # INT8 checks
        int8 = result["int8_tflite"]
        assert int8["predicted_label"] in result["class_names"]
        assert 0.0 <= int8["confidence"] <= 1.0
        assert int8["latency_ms"] > 0.0
        assert len(int8["top3"]) == 3
        assert pytest.approx(sum(int8["all_probabilities"]), abs=1e-2) == 1.0

    def test_predict_invalid_audio_path_raises(self) -> None:
        """Verify that predictor raises AudioValidationError for non-existent audio path."""
        predictor = ModelPredictor()
        with pytest.raises(AudioValidationError, match="does not exist"):
            predictor.predict("data/sample_audio/non_existent_file_12345.wav")


class TestStage8ReliabilityAndMetadata:
    """Test suite for Stage 8 reliability, quantization metadata, and honest reporting."""

    def test_quantization_metadata_int8(self) -> None:
        """Verify INT8 model is fully integer-quantized with valid scale and zero point."""
        import tensorflow as tf
        root = get_project_root()
        int8_path = root / "models" / "quantized" / "baseline_dscnn_int8.tflite"
        assert int8_path.is_file()

        interpreter = tf.lite.Interpreter(model_path=str(int8_path))
        interpreter.allocate_tensors()

        in_details = interpreter.get_input_details()[0]
        out_details = interpreter.get_output_details()[0]

        # Check input details
        assert in_details["dtype"] == np.int8
        assert list(in_details["shape"]) == [1, 40, 47, 1]
        scale, zp = in_details["quantization"]
        assert scale > 0.0
        assert isinstance(zp, (int, np.integer))

        # Check output details
        assert out_details["dtype"] == np.int8
        assert list(out_details["shape"]) == [1, 12]
        out_scale, out_zp = out_details["quantization"]
        assert out_scale > 0.0
        assert isinstance(out_zp, (int, np.integer))

        # Verify no float32 tensor weights
        tensor_details = interpreter.get_tensor_details()
        for t in tensor_details:
            assert t["dtype"] in [np.int8, np.int32], f"Unexpected non-integer tensor: {t['name']} of type {t['dtype']}"

    def test_missing_model_files_error_handling(self, tmp_path: Path) -> None:
        """Verify ModelPredictor raises FileNotFoundError when model files are missing."""
        empty_root = tmp_path / "empty_project"
        empty_root.mkdir()
        predictor = ModelPredictor(project_root=empty_root)
        with pytest.raises(FileNotFoundError):
            predictor._load_pytorch_model()
        with pytest.raises(FileNotFoundError):
            predictor._load_tflite_model()

    def test_verified_vs_estimates_distinction(self) -> None:
        """Verify that resource analysis and pipeline stages clearly separate verified results from estimates."""
        from src.dashboard.data_loader import get_pipeline_stages
        root = get_project_root()

        # 1. Resource analysis assumptions
        res = analyze_model_resources(root)
        assert res["status"] == "Verified (Analytical)"
        assumptions = res["assumptions_and_limitations"]
        assert any("not been measured" in a.lower() or "not performed" in a.lower() for a in assumptions), \
            "Assumptions must state physical latency or power has not been measured"

        # 2. Pipeline stages status
        stages = get_pipeline_stages(root)
        stage8 = next(s for s in stages if s["id"] == 8)
        assert "not performed" in stage8["status"].lower() or "pending" in stage8["status"].lower(), \
            "Stage 8 must be marked as Pending Hardware / Not performed"

    def test_label_ordering_exact_match(self) -> None:
        """Verify label ordering in model_labels.h matches DatasetConfig.get_idx_to_class()."""
        from src.audio.config import DatasetConfig
        root = get_project_root()
        labels_header_path = root / "deployment" / "esp32" / "include" / "model_labels.h"
        header_text = labels_header_path.read_text(encoding="utf-8")

        dataset_cfg = DatasetConfig()
        expected_mapping = dataset_cfg.get_idx_to_class()

        for idx, expected_name in expected_mapping.items():
            pattern = rf'LABEL_\w+\s*=\s*{idx}\b'
            assert re.search(pattern, header_text) is not None, \
                f"Enum mapping for index {idx} ({expected_name}) not found in header"

