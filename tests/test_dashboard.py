"""Unit tests for the Edge AI Compression Dashboard components.

Tests data loading resilience, missing artifact handling, size reduction calculations,
and security constraints for downloadable files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dashboard.data_loader import (
    calculate_size_reduction,
    get_artifact_inventory,
    get_dataset_metadata,
    get_pipeline_stages,
    get_project_root,
    get_safe_downloadable_artifacts,
    safe_read_csv,
    safe_read_json,
    safe_read_text,
)


class TestDashboardDataLoader:
    """Test suite for data loading and calculations in dashboard."""

    def test_get_project_root_resolves_correctly(self) -> None:
        root = get_project_root()
        assert root.is_dir(), "Project root must be a directory"
        assert (root / "configs").is_dir(), "Configs directory must exist in project root"
        assert (root / "src").is_dir(), "Src directory must exist in project root"

    def test_safe_read_json_valid(self) -> None:
        root = get_project_root()
        valid_path = root / "results" / "reports" / "quantization_comparison.json"
        data = safe_read_json(valid_path)
        assert data is not None, "Failed to read existing quantization_comparison.json"
        assert "models" in data
        assert "comparison" in data

    def test_safe_read_json_missing(self) -> None:
        missing_path = get_project_root() / "results" / "reports" / "non_existent_file.json"
        data = safe_read_json(missing_path)
        assert data is None, "Missing file should return None gracefully"

    def test_safe_read_json_corrupted(self, tmp_path: Path) -> None:
        corrupted_file = tmp_path / "corrupted.json"
        corrupted_file.write_text("{invalid json format: missing quotes", encoding="utf-8")
        data = safe_read_json(corrupted_file)
        assert data is None, "Corrupted JSON should return None without raising an exception"

    def test_safe_read_csv_valid_and_missing(self) -> None:
        root = get_project_root()
        valid_csv = root / "results" / "reports" / "training_log.csv"
        df = safe_read_csv(valid_csv)
        assert df is not None, "Valid CSV should return a DataFrame"
        assert "epoch" in df.columns
        assert len(df) == 8, "Expected 8 training epochs in training_log.csv"

        missing_csv = root / "results" / "reports" / "missing_log.csv"
        assert safe_read_csv(missing_csv) is None

    def test_safe_read_text_valid_and_missing(self) -> None:
        root = get_project_root()
        valid_text = root / "results" / "reports" / "operator_compatibility_report.md"
        content = safe_read_text(valid_text)
        assert content is not None
        assert "TFLite Micro" in content

        missing_text = root / "results" / "reports" / "missing_report.md"
        assert safe_read_text(missing_text) is None


class TestSizeReductionCalculations:
    """Test suite for size reduction calculations and boundary conditions."""

    def test_standard_reduction_calculation(self) -> None:
        res = calculate_size_reduction(100.0, 25.0)
        assert res["reduction_bytes"] == 75.0
        assert res["reduction_pct"] == 75.0
        assert res["compression_ratio"] == 4.0
        assert res["status"] == "Valid"

    def test_exact_project_metrics_reduction(self) -> None:
        # PyTorch Checkpoint: 105,396 Bytes -> INT8 TFLite: 22,296 Bytes
        res = calculate_size_reduction(105396, 22296)
        assert res["status"] == "Valid"
        assert res["reduction_pct"] > 78.0
        assert res["compression_ratio"] > 4.7

    def test_edge_case_none_values(self) -> None:
        res1 = calculate_size_reduction(None, 20.0)
        assert res1["status"] == "Not measured"
        assert res1["reduction_pct"] == 0.0

        res2 = calculate_size_reduction(100.0, None)
        assert res2["status"] == "Not measured"

    def test_edge_case_zero_and_negative(self) -> None:
        res_zero = calculate_size_reduction(0, 10)
        assert res_zero["status"] == "Invalid magnitude"

        res_neg = calculate_size_reduction(100, -5)
        assert res_neg["status"] == "Invalid magnitude"

    def test_edge_case_invalid_types(self) -> None:
        res_str = calculate_size_reduction("abc", "def")  # type: ignore
        assert res_str["status"] == "Invalid format"


class TestPipelineMetadataAndSecurity:
    """Test suite for pipeline stage auditing and download path safety."""

    def test_pipeline_stages_structure(self) -> None:
        stages = get_pipeline_stages()
        assert len(stages) == 8, "Expected 8 total pipeline stages"
        for s in stages:
            assert "id" in s
            assert "name" in s
            assert "status" in s
            assert s["status"] in ["Completed", "Pending", "Pending Hardware (Not performed)"]

        # Stages 1 to 7 should be completed
        completed_ids = [s["id"] for s in stages if s["status"] == "Completed"]
        assert 1 in completed_ids
        assert 4 in completed_ids
        assert 5 in completed_ids
        assert 6 in completed_ids
        assert 7 in completed_ids

    def test_artifact_inventory_integrity(self) -> None:
        inventory = get_artifact_inventory()
        assert len(inventory) >= 15, "Expected at least 15 tracked artifacts"
        for item in inventory:
            assert "name" in item
            assert "relative_path" in item
            assert "exists" in item
            assert isinstance(item["exists"], bool)

    def test_dataset_metadata(self) -> None:
        meta = get_dataset_metadata()
        assert meta["num_classes"] == 12
        assert len(meta["target_words"]) == 10
        assert meta["sample_rate"] == 16000
        assert meta["test_samples"] == 518

    def test_safe_downloadable_artifacts_security(self) -> None:
        downloads = get_safe_downloadable_artifacts()
        for key, info in downloads.items():
            rel_path = info["rel_path"]
            # Enforce path confinement to approved directories
            assert (
                rel_path.startswith("results/reports/")
                or rel_path.startswith("deployment/esp32/")
                or rel_path.startswith("models/quantized/")
            ), f"Path {rel_path} escapes approved directories"
            assert ".." not in rel_path, f"Path traversal detected in {rel_path}"
            assert Path(rel_path).suffix in [
                ".json",
                ".csv",
                ".md",
                ".h",
                ".tflite",
            ], f"Untrusted extension in {rel_path}"

