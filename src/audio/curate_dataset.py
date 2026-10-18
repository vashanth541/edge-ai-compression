"""
Dataset Downloader & Curator for Real Google Speech Commands Audio.

Downloads the official Google Speech Commands v0.02 benchmark archive:
URL: http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz
Size: ~107.3 MB compressed, ~156 MB extracted (4,890 WAV files).

Classes Included (12 Total):
- 10 Target Commands: 'down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes'
- 1 Genuine Silence Class: '_silence_' (pre-sliced 1.0s background noise clips)
- 1 Genuine Unknown Class: '_unknown_' (real speech clips of other non-target words)
"""

import argparse
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
import urllib.request
from typing import Dict, Optional

# Ensure workspace root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

OFFICIAL_SPEECH_COMMANDS_BENCHMARK_URL = (
    "http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz"
)


def download_with_progress(url: str, dest_file: Path) -> None:
    """Download a file over HTTP with console progress reporting."""
    print(f"[INFO] Downloading {url} -> {dest_file}...")
    req = urllib.request.Request(url, headers={"User-Agent": "EdgeAI-Compression-Pipeline"})

    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        chunk_size = 1024 * 512  # 512 KB
        downloaded = 0
        t0 = time.time()

        with open(dest_file, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    elapsed = max(time.time() - t0, 0.001)
                    speed_mb = (downloaded / (1024 ** 2)) / elapsed
                    print(
                        f"\r  Downloaded: {downloaded/(1024**2):.1f}/{total_size/(1024**2):.1f} MB ({pct:.1f}%) @ {speed_mb:.2f} MB/s",
                        end="",
                        flush=True,
                    )
        print()


def setup_real_speech_commands(
    dest_dir: str = "data/raw/speech_commands",
    max_samples_per_class: Optional[int] = None,
    keep_archive: bool = False,
) -> Path:
    """
    Download and extract real Google Speech Commands benchmark audio.

    Args:
        dest_dir: Target directory under data/raw/.
        max_samples_per_class: Optional ceiling on samples extracted per class (None for all ~400/class).
        keep_archive: Whether to retain the downloaded .tar.gz archive.

    Returns:
        Path to extracted dataset directory.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    # Check available disk space
    _, _, free_bytes = shutil.disk_usage(dest)
    free_gb = free_bytes / (1024 ** 3)
    if free_gb < 1.0:
        raise RuntimeError(f"Insufficient disk space: {free_gb:.2f} GB free. At least 1.0 GB required.")

    archive_path = dest.parent / "speech_commands_v0.02_benchmark.tar.gz"

    # Step 1: Download archive if not already downloaded
    if not archive_path.exists() or archive_path.stat().st_size < 100 * 1024 * 1024:
        download_with_progress(OFFICIAL_SPEECH_COMMANDS_BENCHMARK_URL, archive_path)

    print(f"[INFO] Extracting archive to {dest}...")
    class_counts: Dict[str, int] = {}

    with tarfile.open(archive_path, mode="r:gz") as tar:
        for member in tar:
            norm_name = member.name.replace("\\", "/")
            if norm_name.startswith("./"):
                norm_name = norm_name[2:]

            parts = norm_name.split("/")
            if len(parts) < 2 or not parts[1].endswith(".wav"):
                continue

            folder = parts[0]
            current_count = class_counts.get(folder, 0)
            if max_samples_per_class is not None and current_count >= max_samples_per_class:
                continue

            tar.extract(member, path=dest)
            class_counts[folder] = current_count + 1

    if not keep_archive and archive_path.exists():
        archive_path.unlink()
        print(f"[INFO] Removed temporary archive {archive_path.name} to save disk space.")

    # Audit extracted files
    total_files = sum(1 for _ in dest.glob("*/*.wav"))
    total_size_mb = sum(f.stat().st_size for f in dest.glob("**/*") if f.is_file()) / (1024 ** 2)

    print("\n=== Real Speech Commands Dataset Summary ===")
    print(f"Directory: {dest.resolve()}")
    print(f"Total Audio Files: {total_files}")
    print(f"Total Storage Used: {total_size_mb:.2f} MB")
    for w in sorted(class_counts.keys()):
        w_count = len(list((dest / w).glob("*.wav"))) if (dest / w).exists() else 0
        print(f"  - {w:20s}: {w_count:4d} files")

    return dest


def main():
    parser = argparse.ArgumentParser(description="Download Real Google Speech Commands Dataset")
    parser.add_argument("--dest_dir", type=str, default="data/raw/speech_commands")
    parser.add_argument("--samples_per_class", type=int, default=None, help="Limit samples per class (default: all ~400)")
    parser.add_argument("--keep_archive", action="store_true", help="Keep .tar.gz archive after extraction")

    args = parser.parse_args()
    setup_real_speech_commands(
        dest_dir=args.dest_dir,
        max_samples_per_class=args.samples_per_class,
        keep_archive=args.keep_archive,
    )


if __name__ == "__main__":
    main()
