"""Standalone PNG/file CLI for deterministic SDF x4 upscaling."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, PngImagePlugin

from . import (
    ADAPTIVE_COLOR_SDF_PIPELINE_ID,
    SAL_CLASSIC_PIPELINE_ID,
    __version__,
    upscale_adaptive_sdf_x4,
    upscale_alpha_sdf_x4,
    upscale_color_sdf_x4,
)

_IMAGE_EXTENSIONS = {".png", ".webp", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}


def _discover(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    )


def _alpha_input(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        if "A" in image.getbands():
            return np.asarray(image.getchannel("A"), dtype=np.uint8)
        return np.asarray(image.convert("L"), dtype=np.uint8)


def _rgba_input(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(ImageOps.exif_transpose(source).convert("RGBA"), dtype=np.uint8)


def _save_png(array: np.ndarray, output: Path, *, pipeline: str, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace it: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("sdf-upscaler.version", __version__)
    metadata.add_text("sdf-upscaler.pipeline", pipeline)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    mode = "L" if array.ndim == 2 else "RGBA"
    Image.fromarray(array, mode=mode).save(temporary, format="PNG", pnginfo=metadata)
    temporary.replace(output)


def _options(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, float | int]]:
    color_options: dict[str, object] = {
        "strength": args.color_strength,
        "edge_band_hr": args.color_edge_band,
    }
    adaptive_options: dict[str, float | int] = {
        "structure_strength": args.adaptive_structure_strength,
        "structure_max_residual": args.adaptive_structure_max_residual,
        "median_max_residual": args.adaptive_median_max_residual,
        "backprojection_iterations": args.adaptive_backprojection_iterations,
        "backprojection_strength": args.adaptive_backprojection_strength,
    }
    return color_options, adaptive_options


def upscale_file(
    source: Path,
    output: Path,
    *,
    backend: str,
    overwrite: bool,
    color_options: dict[str, object],
    adaptive_options: dict[str, float | int],
) -> None:
    """Upscale one image into a PNG using the selected standalone backend."""

    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if backend == "alpha":
        alpha = upscale_alpha_sdf_x4(_alpha_input(source))
        alpha_u8 = np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
        _save_png(alpha_u8, output, pipeline="alpha_sdf_x4_main", overwrite=overwrite)
    elif backend == "color":
        rgba, _ = upscale_color_sdf_x4(_rgba_input(source), rgb_options=color_options)
        _save_png(rgba, output, pipeline=SAL_CLASSIC_PIPELINE_ID, overwrite=overwrite)
    elif backend == "adaptive":
        rgba, _ = upscale_adaptive_sdf_x4(
            _rgba_input(source),
            adaptive_options=adaptive_options,
            color_options=color_options,
        )
        _save_png(rgba, output, pipeline=ADAPTIVE_COLOR_SDF_PIPELINE_ID, overwrite=overwrite)
    else:
        raise ValueError(f"unknown backend: {backend}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="single input image")
    parser.add_argument(
        "--dir",
        dest="input_dir",
        type=Path,
        help="recursively process a directory",
    )
    parser.add_argument("-o", "--output", required=True, type=Path, help="output PNG or directory")
    parser.add_argument("--backend", choices=("alpha", "color", "adaptive"), default="adaptive")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--color-strength", type=float, default=1.0)
    parser.add_argument("--color-edge-band", type=float, default=3.5)
    parser.add_argument("--adaptive-structure-strength", type=float, default=1.0)
    parser.add_argument("--adaptive-structure-max-residual", type=float, default=0.02)
    parser.add_argument("--adaptive-median-max-residual", type=float, default=0.02)
    parser.add_argument("--adaptive-backprojection-iterations", type=int, default=2)
    parser.add_argument("--adaptive-backprojection-strength", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.input is None) == (args.input_dir is None):
        raise ValueError("provide exactly one input image or --dir")
    color_options, adaptive_options = _options(args)
    if args.input is not None:
        upscale_file(
            args.input,
            args.output,
            backend=args.backend,
            overwrite=args.overwrite,
            color_options=color_options,
            adaptive_options=adaptive_options,
        )
        print(args.output.expanduser().resolve())
        return 0

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    inputs = _discover(input_dir)
    if not inputs:
        raise FileNotFoundError(f"no supported images under {input_dir}")
    for position, source in enumerate(inputs, start=1):
        relative = source.relative_to(input_dir).with_suffix(".png")
        destination = output_dir / relative
        upscale_file(
            source,
            destination,
            backend=args.backend,
            overwrite=args.overwrite,
            color_options=color_options,
            adaptive_options=adaptive_options,
        )
        print(f"[{position}/{len(inputs)}] {source} -> {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
