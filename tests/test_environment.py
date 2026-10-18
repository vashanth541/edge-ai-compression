"""
test_environment.py - Stage 1 Environment Smoke Test

Verifies that the Python runtime meets all project specifications:
1. CPython version 3.11.x
2. 64-bit architecture
3. Workspace path write permissions
4. Key standard libraries available
"""

import os
import platform
import struct
import sys
import tempfile
import unittest
from pathlib import Path


class TestStage1Environment(unittest.TestCase):
    def test_python_version(self):
        """Verify Python version is 3.11.x for maximum ML package stability."""
        major = sys.version_info.major
        minor = sys.version_info.minor
        self.assertEqual(major, 3, f"Expected Python 3, got Python {major}")
        self.assertEqual(
            minor, 11, f"Expected Python 3.11 for ML stability, got {major}.{minor}"
        )

    def test_architecture_64bit(self):
        """Verify the interpreter is 64-bit (required for modern ML wheels)."""
        bit_depth = struct.calcsize("P") * 8
        self.assertEqual(bit_depth, 64, f"Expected 64-bit Python, got {bit_depth}-bit")

    def test_virtual_environment(self):
        """Verify execution is within the project virtual environment (.venv)."""
        exe_path = Path(sys.executable).resolve()
        is_in_venv = (
            sys.prefix != sys.base_prefix
            or ".venv" in str(exe_path).lower()
        )
        self.assertTrue(
            is_in_venv,
            f"Expected execution inside virtual environment, but running from: {exe_path}",
        )

    def test_workspace_write_permissions(self):
        """Verify workspace allows file creation and deletion."""
        workspace_root = Path(__file__).resolve().parent.parent
        test_file = workspace_root / ".env_write_test.tmp"
        try:
            test_file.write_text("stage_1_ok", encoding="utf-8")
            self.assertTrue(test_file.exists())
            content = test_file.read_text(encoding="utf-8")
            self.assertEqual(content, "stage_1_ok")
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_standard_libraries(self):
        """Verify essential standard library modules are functional."""
        modules = ["math", "json", "pathlib", "struct", "unittest", "csv", "shutil"]
        for mod in modules:
            with self.subTest(module=mod):
                __import__(mod)


if __name__ == "__main__":
    unittest.main()
