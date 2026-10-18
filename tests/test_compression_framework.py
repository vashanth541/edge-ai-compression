"""Tests for Compression Framework modules.

Covers:
- model_analyzer: valid/invalid TFLite analysis, FlatBuffer validation
- compatibility_checker: check results, error/warning classification
- package_builder: ZIP structure, manifest, C header generation
- dashboard: new Compression Framework section imports
- DS-CNN built-in workflow: analysis of existing model artifacts
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import zipfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_minimal_tflite(identifier: bytes = b"TFL3") -> bytes:
    """
    Craft a minimal valid TFLite FlatBuffer header (8 bytes).
    The first 4 bytes are the FlatBuffer root offset; the next 4 are the identifier.
    The interpreter will reject it as malformed schema, but the magic check passes.
    """
    root_offset = struct.pack("<I", 8)  # offset points to byte 8 (past the header)
    return root_offset + identifier + b"\x00" * 64  # minimal padding


def _load_real_tflite(variant: str = "int8") -> bytes:
    """Load the actual built-in TFLite model bytes."""
    fname = f"baseline_dscnn_{variant}.tflite"
    path = PROJECT_ROOT / "models" / "quantized" / fname
    if not path.is_file():
        pytest.skip(f"Model file not found: {path}")
    with open(path, "rb") as f:
        return f.read()


# ===========================================================================
# A. model_analyzer tests
# ===========================================================================
class TestModelAnalyzerValidation:
    """FlatBuffer validation and file-level checks."""

    def test_valid_flatbuffer_magic_tfl3(self):
        from src.dashboard.model_analyzer import TFLiteModelAnalyzer
        import tempfile, os
        data = _make_minimal_tflite(b"TFL3")
        with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as f:
            f.write(data)
            tmp = f.name
        try:
            # Should not raise on magic check (interpreter may fail later)
            analyzer = TFLiteModelAnalyzer(tmp)
            assert analyzer.path.suffix == ".tflite"
        finally:
            os.unlink(tmp)

    def test_invalid_flatbuffer_magic_raises(self):
        from src.dashboard.model_analyzer import TFLiteAnalysisError, TFLiteModelAnalyzer
        import tempfile, os
        data = _make_minimal_tflite(b"XXXX")  # invalid identifier
        with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as f:
            f.write(data)
            tmp = f.name
        try:
            with pytest.raises(TFLiteAnalysisError, match="identifier"):
                TFLiteModelAnalyzer(tmp)
        finally:
            os.unlink(tmp)

    def test_file_not_found_raises(self):
        from src.dashboard.model_analyzer import TFLiteAnalysisError, TFLiteModelAnalyzer
        with pytest.raises(TFLiteAnalysisError, match="not found"):
            TFLiteModelAnalyzer("/nonexistent/path/model.tflite")

    def test_too_small_file_raises(self):
        from src.dashboard.model_analyzer import TFLiteAnalysisError, TFLiteModelAnalyzer
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as f:
            f.write(b"\x00\x01")  # only 2 bytes
            tmp = f.name
        try:
            with pytest.raises(TFLiteAnalysisError):
                TFLiteModelAnalyzer(tmp)
        finally:
            os.unlink(tmp)


class TestAnalyzeTfliteBytes:
    """analyze_tflite_bytes() public API."""

    def test_empty_bytes_raises(self):
        from src.dashboard.model_analyzer import TFLiteAnalysisError, analyze_tflite_bytes
        with pytest.raises(TFLiteAnalysisError, match="empty"):
            analyze_tflite_bytes(b"", "empty.tflite")

    def test_oversized_raises(self):
        from src.dashboard.model_analyzer import TFLiteAnalysisError, analyze_tflite_bytes
        huge = b"\x00" * (51 * 1024 * 1024)  # 51 MB
        with pytest.raises(TFLiteAnalysisError, match="50 MB"):
            analyze_tflite_bytes(huge, "huge.tflite")

    def test_invalid_magic_raises(self):
        from src.dashboard.model_analyzer import TFLiteAnalysisError, analyze_tflite_bytes
        bad = _make_minimal_tflite(b"BAAD")
        with pytest.raises(TFLiteAnalysisError):
            analyze_tflite_bytes(bad, "bad.tflite")

    def test_real_int8_model_basic_fields(self):
        """Real INT8 model must return correct file-level fields."""
        from src.dashboard.model_analyzer import analyze_tflite_bytes
        data = _load_real_tflite("int8")
        result = analyze_tflite_bytes(data, "baseline_dscnn_int8.tflite")

        assert result["file_name"] == "baseline_dscnn_int8.tflite"
        assert result["file_size_bytes"] == len(data)
        assert result["file_size_kb"] == round(len(data) / 1024.0, 2)
        assert result["sha256"] == hashlib.sha256(data).hexdigest()
        assert result["flatbuffer_identifier"] in ("TFL3", "TFL2", "TFL1")

    def test_real_int8_model_interpreter_fields(self):
        """Real INT8 model must expose input/output tensors from interpreter."""
        from src.dashboard.model_analyzer import analyze_tflite_bytes
        data = _load_real_tflite("int8")
        result = analyze_tflite_bytes(data, "baseline_dscnn_int8.tflite")

        assert result["interpreter_analysis"] == "success", \
            f"Interpreter failed: {result.get('interpreter_analysis')}"
        assert len(result["input_tensors"]) >= 1
        assert len(result["output_tensors"]) >= 1
        assert result["input_dtype"] == "int8"
        assert result["output_dtype"] == "int8"
        assert result["is_fully_integer"] is True

    def test_real_int8_model_input_shape(self):
        """INT8 DS-CNN must have 4D input with batch=1."""
        from src.dashboard.model_analyzer import analyze_tflite_bytes
        data = _load_real_tflite("int8")
        result = analyze_tflite_bytes(data, "baseline_dscnn_int8.tflite")
        shape = result["input_tensors"][0]["shape"]
        assert len(shape) == 4, f"Expected 4D input, got: {shape}"
        assert shape[0] == 1, f"Expected batch=1, got shape: {shape}"

    def test_real_fp32_model_not_fully_integer(self):
        """FP32 TFLite must report is_fully_integer=False."""
        from src.dashboard.model_analyzer import analyze_tflite_bytes
        data = _load_real_tflite("fp32")
        result = analyze_tflite_bytes(data, "baseline_dscnn_fp32.tflite")
        assert result["is_fully_integer"] is False

    def test_file_size_calculation(self):
        """file_size_kb must equal file_size_bytes / 1024 rounded to 2 dp."""
        from src.dashboard.model_analyzer import analyze_tflite_bytes
        data = _load_real_tflite("int8")
        result = analyze_tflite_bytes(data, "test.tflite")
        expected_kb = round(len(data) / 1024.0, 2)
        assert result["file_size_kb"] == expected_kb

    def test_tflm_operator_classification(self):
        """All DS-CNN operators should be TFLM-supported."""
        from src.dashboard.model_analyzer import analyze_tflite_bytes
        data = _load_real_tflite("int8")
        result = analyze_tflite_bytes(data, "test.tflite")
        if result["interpreter_analysis"] == "success":
            unsupported = result.get("tflm_unsupported_operators", [])
            assert unsupported == [], \
                f"Unexpected unsupported operators: {unsupported}"

    def test_format_analysis_for_display_keys(self):
        """format_analysis_for_display must include key display fields."""
        from src.dashboard.model_analyzer import analyze_tflite_bytes, format_analysis_for_display
        data = _load_real_tflite("int8")
        result = analyze_tflite_bytes(data, "test.tflite")
        disp = format_analysis_for_display(result)
        for key in ("File Size", "SHA-256", "Input Shape", "Input Dtype",
                    "Output Shape", "Output Dtype", "Fully Integer (INT8 I/O)"):
            assert key in disp, f"Missing key: {key}"


# ===========================================================================
# B. compatibility_checker tests
# ===========================================================================
class TestCompatibilityChecker:
    """ESP32-S3 compatibility check logic."""

    def _make_analysis(self, **overrides) -> Dict[str, Any]:
        """Minimal valid INT8 analysis dict."""
        base = {
            "file_size_bytes": 22296,
            "file_size_kb": 21.77,
            "flatbuffer_identifier": "TFL3",
            "input_tensors": [{"name": "serving_default_input:0",
                                "shape": [1, 40, 47, 1],
                                "dtype": "int8",
                                "quantization": {"scale": 0.4796, "zero_point": 54}}],
            "output_tensors": [{"name": "StatefulPartitionedCall:0",
                                 "shape": [1, 12],
                                 "dtype": "int8",
                                 "quantization": {"scale": 0.1009, "zero_point": 82}}],
            "input_dtype": "int8",
            "output_dtype": "int8",
            "is_fully_integer": True,
            "unique_operators": ["CONV_2D", "DEPTHWISE_CONV_2D", "AVERAGE_POOL_2D",
                                  "FULLY_CONNECTED", "RESHAPE"],
            "tflm_supported_operators": ["CONV_2D", "DEPTHWISE_CONV_2D",
                                          "AVERAGE_POOL_2D", "FULLY_CONNECTED", "RESHAPE"],
            "tflm_unsupported_operators": [],
            "all_operators_tflm_supported": True,
            "interpreter_analysis": "success",
        }
        base.update(overrides)
        return base

    def test_dscnn_int8_passes_all_checks(self):
        from src.dashboard.compatibility_checker import check_esp32s3_compatibility
        analysis = self._make_analysis()
        report = check_esp32s3_compatibility(analysis)
        assert report.static_checks_passed is True
        assert report.model_export_ready is True
        assert report.errors == []

    def test_unsupported_operator_causes_error(self):
        from src.dashboard.compatibility_checker import check_esp32s3_compatibility
        analysis = self._make_analysis(
            unique_operators=["CONV_2D", "UNKNOWN_OP_XYZ"],
            tflm_unsupported_operators=["UNKNOWN_OP_XYZ"],
            all_operators_tflm_supported=False,
        )
        report = check_esp32s3_compatibility(analysis)
        assert report.static_checks_passed is False
        assert any("UNKNOWN_OP_XYZ" in e for e in report.errors)

    def test_invalid_flatbuffer_causes_error(self):
        from src.dashboard.compatibility_checker import check_esp32s3_compatibility
        analysis = self._make_analysis(flatbuffer_identifier="XXXX")
        report = check_esp32s3_compatibility(analysis)
        assert report.static_checks_passed is False

    def test_float_input_generates_warning_not_error(self):
        from src.dashboard.compatibility_checker import check_esp32s3_compatibility
        analysis = self._make_analysis(
            input_dtype="float32",
            output_dtype="float32",
            is_fully_integer=False,
            input_tensors=[{"name": "inp", "shape": [1, 40, 47, 1],
                             "dtype": "float32", "quantization": None}],
            output_tensors=[{"name": "out", "shape": [1, 12],
                              "dtype": "float32", "quantization": None}],
        )
        report = check_esp32s3_compatibility(analysis)
        # float I/O is a warning, not a hard error in TFLM
        float_check = next(
            (c for c in report.checks if "Integer" in c["Check"]), None
        )
        assert float_check is not None
        assert float_check["Status"] in ("⚠️ Warning", "✅ Pass")

    def test_oversized_model_causes_error(self):
        from src.dashboard.compatibility_checker import check_esp32s3_compatibility
        analysis = self._make_analysis(
            file_size_kb=600.0,
            file_size_bytes=600 * 1024,
        )
        report = check_esp32s3_compatibility(analysis)
        assert report.static_checks_passed is False
        assert any("Flash" in e or "KB" in e for e in report.errors)

    def test_firmware_build_always_not_performed(self):
        from src.dashboard.compatibility_checker import check_esp32s3_compatibility
        report = check_esp32s3_compatibility(self._make_analysis())
        assert "not performed" in report.firmware_build_status.lower()
        assert "not performed" in report.hardware_test_status.lower()

    def test_summary_status_pass(self):
        from src.dashboard.compatibility_checker import check_esp32s3_compatibility
        report = check_esp32s3_compatibility(self._make_analysis())
        emoji, label = report.summary_status()
        assert emoji in ("✅", "⚠️", "❌")

    def test_generate_compatibility_markdown(self):
        from src.dashboard.compatibility_checker import (
            check_esp32s3_compatibility,
            generate_compatibility_markdown,
        )
        report = check_esp32s3_compatibility(self._make_analysis())
        md = generate_compatibility_markdown(report, "test_model")
        assert "ESP32-S3" in md
        assert "test_model" in md
        assert "| Check |" in md  # table header
        assert "Not performed" in md

    def test_to_dict_structure(self):
        from src.dashboard.compatibility_checker import check_esp32s3_compatibility
        report = check_esp32s3_compatibility(self._make_analysis())
        d = report.to_dict()
        for key in ("target_profile", "checks", "warnings", "errors",
                    "model_export_ready", "static_checks_passed",
                    "firmware_build_status", "hardware_test_status"):
            assert key in d, f"Missing key: {key}"


# ===========================================================================
# C. package_builder tests
# ===========================================================================
class TestPackageBuilder:
    """ZIP package construction and manifest integrity."""

    def _build_package(self, variant="int8") -> bytes:
        from src.dashboard.model_analyzer import analyze_tflite_bytes
        from src.dashboard.compatibility_checker import (
            check_esp32s3_compatibility, generate_compatibility_markdown
        )
        from src.dashboard.package_builder import build_deployment_package

        data = _load_real_tflite(variant)
        fname = f"baseline_dscnn_{variant}.tflite"
        analysis = analyze_tflite_bytes(data, fname)
        report = check_esp32s3_compatibility(analysis)
        compat_md = generate_compatibility_markdown(report, f"baseline_dscnn_{variant}")
        return build_deployment_package(
            tflite_bytes=data,
            tflite_filename=fname,
            analysis=analysis,
            compatibility_report_md=compat_md,
            json_report=report.to_dict(),
        )

    def test_package_is_valid_zip(self):
        pkg = self._build_package()
        assert zipfile.is_zipfile(io.BytesIO(pkg))

    def test_package_contains_required_files(self):
        pkg = self._build_package()
        with zipfile.ZipFile(io.BytesIO(pkg)) as zf:
            names = set(zf.namelist())
        required = {
            "include/model_data.h",
            "include/model_labels.h",
            "main.cpp",
            "compatibility_report.md",
            "deployment_report.json",
            "README.md",
            "MANIFEST.json",
        }
        for r in required:
            assert r in names, f"Missing from ZIP: {r}"

    def test_package_contains_tflite_model(self):
        pkg = self._build_package()
        with zipfile.ZipFile(io.BytesIO(pkg)) as zf:
            names = set(zf.namelist())
        tflite_files = [n for n in names if n.endswith(".tflite")]
        assert len(tflite_files) == 1

    def test_tflite_bytes_unchanged_in_package(self):
        data = _load_real_tflite("int8")
        from src.dashboard.model_analyzer import analyze_tflite_bytes
        from src.dashboard.compatibility_checker import (
            check_esp32s3_compatibility, generate_compatibility_markdown
        )
        from src.dashboard.package_builder import build_deployment_package
        fname = "baseline_dscnn_int8.tflite"
        analysis = analyze_tflite_bytes(data, fname)
        report = check_esp32s3_compatibility(analysis)
        pkg = build_deployment_package(
            tflite_bytes=data,
            tflite_filename=fname,
            analysis=analysis,
            compatibility_report_md=generate_compatibility_markdown(report, "test"),
            json_report=report.to_dict(),
        )
        with zipfile.ZipFile(io.BytesIO(pkg)) as zf:
            packed = zf.read(f"models/{fname}")
        assert packed == data, "TFLite bytes were modified inside the package"

    def test_c_header_contains_alignas(self):
        pkg = self._build_package()
        with zipfile.ZipFile(io.BytesIO(pkg)) as zf:
            header = zf.read("include/model_data.h").decode()
        assert "alignas(8)" in header

    def test_c_header_contains_array_name(self):
        pkg = self._build_package()
        with zipfile.ZipFile(io.BytesIO(pkg)) as zf:
            header = zf.read("include/model_data.h").decode()
        assert "g_model_data" in header
        assert "g_model_data_len" in header

    def test_labels_header_contains_class_names(self):
        pkg = self._build_package()
        with zipfile.ZipFile(io.BytesIO(pkg)) as zf:
            labels_h = zf.read("include/model_labels.h").decode()
        assert "kClassLabels" in labels_h
        assert "kNumClasses" in labels_h

    def test_deployment_report_json_is_valid(self):
        pkg = self._build_package()
        with zipfile.ZipFile(io.BytesIO(pkg)) as zf:
            raw = zf.read("deployment_report.json")
        report = json.loads(raw)
        for key in ("generated_at", "model_name", "file_size_bytes",
                    "sha256", "deployment_status"):
            assert key in report, f"Missing key: {key}"

    def test_deployment_status_never_claims_firmware_built(self):
        pkg = self._build_package()
        with zipfile.ZipFile(io.BytesIO(pkg)) as zf:
            report = json.loads(zf.read("deployment_report.json"))
        fw_status = report["deployment_status"]["firmware_build"]
        assert "not_performed" in fw_status.lower() or "not performed" in fw_status.lower()

    def test_manifest_sha256_matches_files(self):
        pkg = self._build_package()
        with zipfile.ZipFile(io.BytesIO(pkg)) as zf:
            manifest = json.loads(zf.read("MANIFEST.json"))
            for entry in manifest["artifacts"]:
                fname = entry["filename"]
                actual = hashlib.sha256(zf.read(fname)).hexdigest()
                assert actual == entry["sha256"], \
                    f"SHA-256 mismatch for {fname}"

    def test_manifest_sizes_match_actual(self):
        pkg = self._build_package()
        with zipfile.ZipFile(io.BytesIO(pkg)) as zf:
            manifest = json.loads(zf.read("MANIFEST.json"))
            for entry in manifest["artifacts"]:
                actual_size = len(zf.read(entry["filename"]))
                assert actual_size == entry["size_bytes"], \
                    f"Size mismatch for {entry['filename']}"

    def test_main_cpp_mentions_template_notice(self):
        pkg = self._build_package()
        with zipfile.ZipFile(io.BytesIO(pkg)) as zf:
            cpp = zf.read("main.cpp").decode()
        assert "TEMPLATE ONLY" in cpp or "template" in cpp.lower()

    def test_readme_contains_disclaimer(self):
        pkg = self._build_package()
        with zipfile.ZipFile(io.BytesIO(pkg)) as zf:
            readme = zf.read("README.md").decode()
        assert "NOT PERFORMED" in readme or "not performed" in readme.lower()

    def test_c_header_length_constant_correct(self):
        """The length constant in the C header must equal actual model bytes."""
        data = _load_real_tflite("int8")
        from src.dashboard.model_analyzer import analyze_tflite_bytes
        from src.dashboard.compatibility_checker import (
            check_esp32s3_compatibility, generate_compatibility_markdown
        )
        from src.dashboard.package_builder import build_deployment_package
        fname = "baseline_dscnn_int8.tflite"
        analysis = analyze_tflite_bytes(data, fname)
        report = check_esp32s3_compatibility(analysis)
        pkg = build_deployment_package(
            tflite_bytes=data,
            tflite_filename=fname,
            analysis=analysis,
            compatibility_report_md=generate_compatibility_markdown(report, "test"),
            json_report=report.to_dict(),
        )
        with zipfile.ZipFile(io.BytesIO(pkg)) as zf:
            header = zf.read("include/model_data.h").decode()
        expected = str(len(data))
        assert expected in header, \
            f"Expected length constant {expected} not found in model_data.h"


# ===========================================================================
# D. DS-CNN built-in model model-size accuracy
# ===========================================================================
class TestDSCNNBuiltinModel:
    """Verify the existing quantized model artifacts are as expected."""

    def test_int8_model_file_size(self):
        """INT8 model must be 22,296 bytes as previously verified."""
        path = PROJECT_ROOT / "models" / "quantized" / "baseline_dscnn_int8.tflite"
        if not path.is_file():
            pytest.skip("INT8 model not found")
        assert path.stat().st_size == 22296, \
            f"Unexpected INT8 model size: {path.stat().st_size}"

    def test_fp32_model_exists(self):
        path = PROJECT_ROOT / "models" / "quantized" / "baseline_dscnn_fp32.tflite"
        if not path.is_file():
            pytest.skip("FP32 TFLite model not found")
        assert path.stat().st_size > 0

    def test_checkpoint_exists(self):
        path = PROJECT_ROOT / "models" / "checkpoints" / "baseline_dscnn.pt"
        if not path.is_file():
            pytest.skip("Checkpoint not found")
        assert path.stat().st_size > 0

    def test_int8_model_fully_integer(self):
        from src.dashboard.model_analyzer import analyze_tflite_file
        path = PROJECT_ROOT / "models" / "quantized" / "baseline_dscnn_int8.tflite"
        if not path.is_file():
            pytest.skip("INT8 model not found")
        result = analyze_tflite_file(path)
        if result.get("interpreter_analysis") == "success":
            assert result["is_fully_integer"] is True

    def test_int8_model_passes_compatibility(self):
        from src.dashboard.model_analyzer import analyze_tflite_file
        from src.dashboard.compatibility_checker import check_esp32s3_compatibility
        path = PROJECT_ROOT / "models" / "quantized" / "baseline_dscnn_int8.tflite"
        if not path.is_file():
            pytest.skip("INT8 model not found")
        analysis = analyze_tflite_file(path)
        if analysis.get("interpreter_analysis") != "success":
            pytest.skip("Interpreter not available for compatibility check")
        report = check_esp32s3_compatibility(analysis)
        assert report.static_checks_passed is True, \
            f"Compatibility check failed: {report.errors}"


# ===========================================================================
# E. Module import smoke test
# ===========================================================================
class TestModuleImports:
    """Verify all new modules import without error."""

    def test_model_analyzer_imports(self):
        from src.dashboard import model_analyzer  # noqa: F401
        assert hasattr(model_analyzer, "analyze_tflite_bytes")
        assert hasattr(model_analyzer, "analyze_tflite_file")
        assert hasattr(model_analyzer, "format_analysis_for_display")

    def test_compatibility_checker_imports(self):
        from src.dashboard import compatibility_checker  # noqa: F401
        assert hasattr(compatibility_checker, "check_esp32s3_compatibility")
        assert hasattr(compatibility_checker, "generate_compatibility_markdown")
        assert hasattr(compatibility_checker, "ESP32S3_PROFILE")

    def test_package_builder_imports(self):
        from src.dashboard import package_builder  # noqa: F401
        assert hasattr(package_builder, "build_deployment_package")

    def test_esp32s3_profile_values(self):
        from src.dashboard.compatibility_checker import ESP32S3_PROFILE
        assert ESP32S3_PROFILE["total_sram_kb"] == 512
        assert ESP32S3_PROFILE["spi_flash_mb"] == 4
        assert ESP32S3_PROFILE["chip"] == "ESP32-S3"


# ===========================================================================
# F. Verified Deployment Package Hero Banner Tests
# ===========================================================================
class TestVerifiedDeploymentPackage:
    """Verify package loader, cache, and integrity validator for Streamlit Cloud."""

    def test_get_verified_package_success(self):
        from src.dashboard.package_builder import get_verified_deployment_package
        info = get_verified_deployment_package(str(PROJECT_ROOT))
        assert info["valid"] is True
        assert info["filename"] == "baseline_dscnn_int8_esp32s3_deploy.zip"
        assert info["model_size_bytes"] == 22296
        assert info["model_size_kb"] == 21.77
        assert info["zip_bytes"] is not None
        assert len(info["zip_bytes"]) > 0

    def test_get_verified_package_missing_path(self):
        from src.dashboard.package_builder import get_verified_deployment_package
        info = get_verified_deployment_package("/nonexistent/directory/path")
        assert info["valid"] is False
        assert info["zip_bytes"] is None
        assert "missing" in info["error"].lower()

    def test_get_verified_package_manifest_integrity(self):
        import io, zipfile, json, hashlib
        from src.dashboard.package_builder import get_verified_deployment_package
        info = get_verified_deployment_package(str(PROJECT_ROOT))
        assert info["valid"] is True
        with zipfile.ZipFile(io.BytesIO(info["zip_bytes"])) as zf:
            manifest = json.loads(zf.read("MANIFEST.json"))
            for item in manifest["artifacts"]:
                data = zf.read(item["filename"])
                actual_hash = hashlib.sha256(data).hexdigest()
                assert actual_hash == item["sha256"], f"Hash mismatch for {item['filename']}"
                assert len(data) == item["size_bytes"], f"Size mismatch for {item['filename']}"

