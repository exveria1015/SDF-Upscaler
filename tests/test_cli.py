from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sdf_upscaler.cli import build_parser, main


def _write_rgba(path: Path) -> None:
    rgba = np.zeros((10, 12, 4), dtype=np.uint8)
    rgba[2:8, 3:10, :3] = (80, 190, 110)
    rgba[2:8, 3:10, 3] = 255
    Image.fromarray(rgba, mode="RGBA").save(path)


def test_parser_defaults_to_adaptive() -> None:
    args = build_parser().parse_args(["input.png", "-o", "output.png"])
    assert args.backend == "adaptive"


@pytest.mark.parametrize(
    ("backend", "mode"),
    (("alpha", "L"), ("color", "RGBA"), ("adaptive", "RGBA")),
)
def test_cli_writes_x4_png(tmp_path: Path, backend: str, mode: str) -> None:
    source = tmp_path / "input.png"
    output = tmp_path / f"{backend}.png"
    _write_rgba(source)

    assert main([str(source), "-o", str(output), "--backend", backend]) == 0

    with Image.open(output) as image:
        assert image.size == (48, 40)
        assert image.mode == mode
        assert image.info["sdf-upscaler.version"] == "0.1.0"


def test_cli_requires_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "input.png"
    output = tmp_path / "output.png"
    _write_rgba(source)
    main([str(source), "-o", str(output), "--backend", "alpha"])
    with pytest.raises(FileExistsError, match="--overwrite"):
        main([str(source), "-o", str(output), "--backend", "alpha"])
