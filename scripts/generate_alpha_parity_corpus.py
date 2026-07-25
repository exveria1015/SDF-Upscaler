"""Generate and verify the deterministic multi-case Python Alpha-SDF parity corpus.

Run with ``uv run python scripts/generate_alpha_parity_corpus.py``.  The
``--verify`` mode leaves fixtures untouched and proves that the checked-in
bytes, dimensions, and hashes still match a fresh canonical Python run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from sdf_upscaler import __version__, main_alpha_x4_options, upscale_alpha_sdf_x4

SCHEMA_VERSION = 1
SCALE = 4
PIPELINE_ID = "python-alpha-sdf/main-alpha-x4/sal-r0-gsupport-scs/v1"
INPUT_FILE = "input_alpha_u8.raw"
FLOAT_FILE = "expected_alpha_f32_le.raw"
U8_FILE = "expected_alpha_u8.raw"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canvas(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype=np.uint8)


def _zero() -> np.ndarray:
    return _canvas(7, 9)


def _one() -> np.ndarray:
    return np.full((7, 9), 255, dtype=np.uint8)


def _broad_soft() -> np.ndarray:
    yy, xx = np.mgrid[:9, :11]
    # No hard endpoints: this exercises the calibrated broad-soft gate.
    return np.rint(28.0 + 199.0 * (0.62 * xx / 10.0 + 0.38 * yy / 8.0)).astype(np.uint8)


def _rectangle() -> np.ndarray:
    alpha = _canvas(9, 13)
    alpha[2:7, 3:10] = 255
    return alpha


def _ellipse() -> np.ndarray:
    yy, xx = np.mgrid[:11, :15]
    return np.where(((xx - 7.0) / 4.8) ** 2 + ((yy - 5.0) / 3.6) ** 2 <= 1.0, 255, 0).astype(
        np.uint8
    )


def _line_horizontal() -> np.ndarray:
    alpha = _canvas(7, 13)
    alpha[3, 2:11] = 255
    return alpha


def _line_vertical() -> np.ndarray:
    alpha = _canvas(13, 7)
    alpha[2:11, 3] = 255
    return alpha


def _line_diagonal() -> np.ndarray:
    alpha = _canvas(11, 11)
    alpha[np.arange(2, 9), np.arange(2, 9)] = 255
    return alpha


def _holes() -> np.ndarray:
    alpha = _canvas(13, 15)
    alpha[2:11, 2:13] = 255
    alpha[5:8, 5:10] = 0
    return alpha


def _open_contour() -> np.ndarray:
    alpha = _canvas(13, 15)
    # A one-pixel C stroke: deliberately open rather than a filled silhouette.
    alpha[2:11, 3] = 255
    alpha[2, 3:11] = 255
    alpha[10, 3:11] = 255
    return alpha


def _partial_alpha_edges() -> np.ndarray:
    alpha = _canvas(9, 13)
    alpha[2:7, 3:10] = 255
    alpha[1, 3:10] = 64
    alpha[7, 3:10] = 192
    alpha[2:7, 2] = 96
    alpha[2:7, 10] = 160
    alpha[1, 2] = 32
    alpha[1, 10] = 48
    alpha[7, 2] = 208
    alpha[7, 10] = 224
    return alpha


def _odd_size_motif() -> np.ndarray:
    alpha = _canvas(9, 11)
    alpha[1:8, 5] = 255
    alpha[4, 1:10] = 255
    alpha[2, 2] = 128
    alpha[6, 8] = 128
    return alpha


def _scs_lift() -> np.ndarray:
    """Sparse partial-alpha branch whose protected SAL center needs SCS lift."""

    alpha = _canvas(9, 9)
    for (y, x), value in {
        (3, 4): 255,
        (4, 1): 191,
        (4, 3): 191,
        (4, 4): 191,
        (5, 2): 191,
        (5, 4): 153,
        (6, 2): 191,
        (6, 3): 230,
        (6, 4): 130,
        (6, 5): 230,
        (7, 4): 191,
        (7, 5): 255,
        (7, 6): 153,
        (7, 7): 191,
    }.items():
        alpha[y, x] = value
    return alpha


CASES: tuple[tuple[str, str, Callable[[], np.ndarray]], ...] = (
    ("zero", "constant zero alpha", _zero),
    ("one", "constant one alpha", _one),
    ("broad_soft", "broad non-endpoint soft gradient", _broad_soft),
    ("rectangle", "hard filled rectangle", _rectangle),
    ("ellipse", "hard filled ellipse", _ellipse),
    ("line_horizontal", "one-pixel horizontal line art", _line_horizontal),
    ("line_vertical", "one-pixel vertical line art", _line_vertical),
    ("line_diagonal", "one-pixel diagonal line art", _line_diagonal),
    ("holes", "filled region with a rectangular hole", _holes),
    ("open_contour", "open one-pixel C contour", _open_contour),
    ("partial_alpha_edges", "hard rectangle with partial-alpha edge samples", _partial_alpha_edges),
    ("odd_size_motif", "asymmetric odd-sized line-art motif", _odd_size_motif),
    ("scs_lift", "partial-alpha thin branch with active SCS medial lift", _scs_lift),
)


def _versions() -> dict[str, str]:
    return {
        package: importlib.metadata.version(package)
        for package in ("numpy", "pillow", "scipy", "scikit-image")
    } | {"sdf-upscaler": __version__}


def _render(alpha: np.ndarray) -> tuple[np.ndarray, bytes, bytes, bytes]:
    if alpha.dtype != np.uint8 or alpha.ndim != 2:
        raise ValueError("cases must return a two-dimensional uint8 alpha plane")
    expected = upscale_alpha_sdf_x4(alpha).astype("<f4", copy=False)
    expected_u8 = np.rint(np.clip(expected, 0.0, 1.0) * 255.0).astype(np.uint8)
    return expected, alpha.tobytes(), expected.tobytes(), expected_u8.tobytes()


def _case_manifest(
    name: str, description: str, alpha: np.ndarray
) -> tuple[dict[str, object], tuple[bytes, bytes, bytes]]:
    expected, input_bytes, float_bytes, u8_bytes = _render(alpha)
    height, width = alpha.shape
    output_height, output_width = expected.shape
    entry: dict[str, object] = {
        "id": name,
        "description": description,
        "input": {
            "path": f"{name}/{INPUT_FILE}", "width": width, "height": height,
            "dtype": "uint8", "byte_length": len(input_bytes), "sha256": _sha256(input_bytes),
        },
        "output": {
            "width": output_width, "height": output_height,
            "float": {
                "path": f"{name}/{FLOAT_FILE}", "dtype": "little-endian float32",
                "byte_length": len(float_bytes), "sha256": _sha256(float_bytes),
            },
            "uint8": {
                "path": f"{name}/{U8_FILE}", "dtype": "uint8",
                "byte_length": len(u8_bytes), "sha256": _sha256(u8_bytes),
            },
        },
    }
    return entry, (input_bytes, float_bytes, u8_bytes)


def _manifest() -> tuple[dict[str, object], list[tuple[str, tuple[bytes, bytes, bytes]]]]:
    entries = []
    payloads = []
    for name, description, factory in CASES:
        entry, rendered = _case_manifest(name, description, factory())
        entries.append(entry)
        payloads.append((name, rendered))
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "python_alpha_sdf_x4_parity_corpus",
        "pipeline": {"id": PIPELINE_ID, "scale": SCALE, "options": main_alpha_x4_options()},
        "versions": _versions(),
        "cases": entries,
    }, payloads


def generate(output: Path) -> None:
    manifest, payloads = _manifest()
    output.mkdir(parents=True, exist_ok=True)
    for name, (input_bytes, float_bytes, u8_bytes) in payloads:
        case_directory = output / name
        case_directory.mkdir(parents=True, exist_ok=True)
        (case_directory / INPUT_FILE).write_bytes(input_bytes)
        (case_directory / FLOAT_FILE).write_bytes(float_bytes)
        (case_directory / U8_FILE).write_bytes(u8_bytes)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    verify(output)


def verify(output: Path) -> None:
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing manifest: {manifest_path}")
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    regenerated, payloads = _manifest()
    if stored != regenerated:
        raise ValueError("manifest does not match a fresh canonical Python run")
    for entry, (name, payload) in zip(stored["cases"], payloads, strict=True):
        input_bytes, float_bytes, u8_bytes = payload
        if entry["id"] != name:
            raise ValueError(f"manifest case order/id mismatch for {name}")
        for metadata, filename, expected in (
            (entry["input"], INPUT_FILE, input_bytes),
            (entry["output"]["float"], FLOAT_FILE, float_bytes),
            (entry["output"]["uint8"], U8_FILE, u8_bytes),
        ):
            actual = (output / name / filename).read_bytes()
            if len(actual) != metadata["byte_length"] or _sha256(actual) != metadata["sha256"]:
                raise ValueError(f"stored bytes/hash mismatch: {name}/{filename}")
            if actual != expected:
                raise ValueError(f"stored bytes differ from regenerated output: {name}/{filename}")
        input_meta = entry["input"]
        output_meta = entry["output"]
        dimensions_match = (
            output_meta["width"] == input_meta["width"] * SCALE
            and output_meta["height"] == input_meta["height"] * SCALE
        )
        if not dimensions_match:
            raise ValueError(f"declared dimensions are not x{SCALE}: {name}")
        if input_meta["byte_length"] != input_meta["width"] * input_meta["height"]:
            raise ValueError(f"input byte length disagrees with dimensions: {name}")
        pixels = output_meta["width"] * output_meta["height"]
        expected_float_bytes = pixels * 4
        expected_u8_bytes = pixels
        if (
            output_meta["float"]["byte_length"] != expected_float_bytes
            or output_meta["uint8"]["byte_length"] != expected_u8_bytes
        ):
            raise ValueError(f"output byte lengths disagree with dimensions: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("fixtures/parity/alpha_cases"))
    parser.add_argument(
        "--verify", action="store_true", help="verify checked-in corpus without writing"
    )
    args = parser.parse_args()
    if args.verify:
        verify(args.output)
    else:
        generate(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
