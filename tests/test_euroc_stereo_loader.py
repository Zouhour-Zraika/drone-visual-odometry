"""Unit tests for the legacy EuRoC stereo index loader."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from euroc_stereo_loader import load_stereo_index


class StereoLoaderTests(unittest.TestCase):
    def test_valid_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "MH_01_easy.txt.d"
            images.mkdir()
            lines = []
            for side in ("left", "right"):
                name = f"camera_{side}.image_raw_00000000.png"
                (images / name).write_bytes(b"dummy")
                lines.append(
                    f"PINHOLE_IMAGE_MESSAGE /camera_{side}/image_raw "
                    f"camera_{side} 00000000 1403636579.76356 0 0 0 "
                    f"MH_01_easy.txt.d/{name} 435.205\n"
                )
            (root / "MH_01_easy.txt").write_text("".join(lines))
            frames = load_stereo_index(root)
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0].index, 0)

    def test_missing_partner(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "MH_01_easy.txt.d"
            images.mkdir()
            name = "camera_left.image_raw_00000000.png"
            (images / name).write_bytes(b"dummy")
            (root / "MH_01_easy.txt").write_text(
                "PINHOLE_IMAGE_MESSAGE /camera_left/image_raw "
                f"camera_left 00000000 1403636579.76356 "
                f"MH_01_easy.txt.d/{name}\n"
            )
            with self.assertRaisesRegex(ValueError, "Missing stereo partner"):
                load_stereo_index(root)


if __name__ == "__main__":
    unittest.main()
