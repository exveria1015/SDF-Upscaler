"""Generate the canonical Python Alpha-SDF fixture consumed by Rust tests."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from sdf_upscaler import upscale_alpha_sdf_x4


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def generate(output: Path) -> None:
    height, width = 18, 22
    yy, xx = np.mgrid[:height, :width]
    alpha = np.where(
        ((xx - 10.5) / 7.5) ** 2 + ((yy - 8.5) / 6.0) ** 2 <= 1.0,
        255,
        0,
    ).astype(np.uint8)
    alpha[3, 6:16] = 128
    expected = upscale_alpha_sdf_x4(alpha).astype("<f4", copy=False)
    expected_u8 = np.rint(np.clip(expected, 0.0, 1.0) * 255.0).astype(np.uint8)

    output.mkdir(parents=True, exist_ok=True)
    input_bytes = alpha.tobytes()
    float_bytes = expected.tobytes()
    u8_bytes = expected_u8.tobytes()
    (output / "input_alpha_u8.raw").write_bytes(input_bytes)
    (output / "expected_alpha_f32_le.raw").write_bytes(float_bytes)
    (output / "expected_alpha_u8.raw").write_bytes(u8_bytes)
    manifest = {
        "schema_version": 1,
        "kind": "python_alpha_sdf_x4_golden",
        "input": {
            "width": width,
            "height": height,
            "dtype": "uint8",
            "sha256": _sha256(input_bytes),
        },
        "output": {
            "width": width * 4,
            "height": height * 4,
            "float_dtype": "little-endian float32",
            "float_sha256": _sha256(float_bytes),
            "uint8_sha256": _sha256(u8_bytes),
        },
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "pillow", "scipy", "scikit-image", "sdf-upscaler")
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fixtures/parity/alpha"),
    )
    args = parser.parse_args()
    generate(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
