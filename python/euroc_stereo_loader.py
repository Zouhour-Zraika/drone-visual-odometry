"""Read paired stereo frames from the legacy EuRoC MH_01_easy.txt archive.

This format is a proSLAM-style export, not the official EuRoC MAV CSV layout.
No IMU is present in the observed archive. Ground truth is never used here.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class StereoFrame:
    index: int
    timestamp_s: float
    left: Path
    right: Path


def load_stereo_index(root: Path) -> list[StereoFrame]:
    root = Path(root)
    manifest = root / "MH_01_easy.txt"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest}")

    grouped: dict[int, dict[str, tuple[float, Path]]] = {}
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) < 5 or fields[0] != "PINHOLE_IMAGE_MESSAGE":
                raise ValueError(f"Unexpected record at line {line_number}")
            topic, camera = fields[1:3]
            if (topic, camera) not in {
                ("/camera_left/image_raw", "camera_left"),
                ("/camera_right/image_raw", "camera_right"),
            }:
                raise ValueError(f"Unexpected camera at line {line_number}: {topic}")
            index = int(fields[3])
            timestamp = float(fields[4])
            # Paths in this export are relative to the manifest directory.
            candidates = [token for token in fields[5:] if token.endswith(".png")]
            if len(candidates) != 1:
                raise ValueError(f"Expected one PNG path at line {line_number}")
            relative = Path(candidates[0])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe image path at line {line_number}")
            image_path = root / relative
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing image at line {line_number}: {image_path}")
            entry = grouped.setdefault(index, {})
            if camera in entry:
                raise ValueError(f"Duplicate {camera} frame {index}")
            entry[camera] = (timestamp, image_path)

    if not grouped:
        raise ValueError("No camera records found")

    frames = []
    for index, pair in sorted(grouped.items()):
        if set(pair) != {"camera_left", "camera_right"}:
            raise ValueError(f"Missing stereo partner at frame {index}")
        left_t, left_path = pair["camera_left"]
        right_t, right_path = pair["camera_right"]
        if abs(left_t - right_t) > 0.001:
            raise ValueError(f"Stereo timestamp mismatch at frame {index}")
        frames.append(StereoFrame(index, left_t, left_path, right_path))
    if any(b.timestamp_s <= a.timestamp_s for a, b in zip(frames, frames[1:])):
        raise ValueError("Non-increasing frame timestamps")
    return frames


def read_pair(frame: StereoFrame):
    left = cv2.imread(str(frame.left), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(frame.right), cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise ValueError(f"Unable to decode stereo frame {frame.index}")
    if left.shape != right.shape:
        raise ValueError(f"Stereo image size mismatch at frame {frame.index}")
    return left, right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/euroc/MH_01_easy"))
    parser.add_argument("--check-all", action="store_true",
                        help="Decode every stereo pair (slower)")
    args = parser.parse_args()
    frames = load_stereo_index(args.root)
    selected = frames if args.check_all else [frames[0], frames[-1]]
    for frame in selected:
        left, right = read_pair(frame)
    print(f"Stereo pairs: {len(frames)}")
    print(f"First/last frame: {frames[0].index} / {frames[-1].index}")
    print(f"Duration: {frames[-1].timestamp_s - frames[0].timestamp_s:.3f} s")
    print(f"Image shape: {left.shape[1]}x{left.shape[0]}")
    print(f"Decoded pairs: {len(selected)}")
    print("No camera calibration or IMU fusion performed.")


if __name__ == "__main__":
    main()
