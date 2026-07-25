"""Generate and verify deterministic Color-SDF / SAL-Classic parity fixtures."""

# ruff: noqa: E501, E701, E702

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from sdf_upscaler import alpha_sdf, main_sal_classic_rgb_options, sal_classic, upscale_color_sdf_x4
from sdf_upscaler.rgb_gate import _srgb_to_linear, resize_premultiplied_rgb_lanczos

OUTPUT = Path("fixtures/parity/color")
SCALE = 4
EXPECTED_FIXED_RGBA_SHA256 = "f00ce3c04297942c59ad8b7986cb781151a8a3a22374ce1d1484e8beee10219c"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture() -> np.ndarray:
    height, width = 18, 22
    yy, xx = np.mgrid[:height, :width]
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = np.stack(((xx * 17 + yy * 3) % 256, (xx * 7 + yy * 19) % 256, (xx * 13 + yy * 11) % 256), axis=2).astype(np.uint8)
    mask = ((xx - 10.5) / 7.5) ** 2 + ((yy - 8.5) / 6.0) ** 2 <= 1.0
    rgba[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
    rgba[3, 6:16, 3] = 128
    return rgba


def _opaque_flat() -> np.ndarray:
    rgba = np.zeros((5, 7, 4), dtype=np.uint8); rgba[..., :3] = (20, 130, 220); rgba[..., 3] = 255; return rgba


def _hard_chroma_split() -> np.ndarray:
    rgba = _opaque_flat(); rgba[:, :3, :3] = (240, 30, 20); rgba[:, 3:, :3] = (20, 30, 240); return rgba


def _partial_edge() -> np.ndarray:
    rgba = np.zeros((7, 9, 4), dtype=np.uint8); rgba[..., :3] = (255, 80, 10); rgba[2:5, 2:7, 3] = 255; rgba[1, 2:7, 3] = 96; rgba[5, 2:7, 3] = 192; return rgba


def _hidden_rgb(hidden: tuple[int, int, int]) -> np.ndarray:
    rgba = _partial_edge(); rgba[..., :3][rgba[..., 3] == 0] = hidden; return rgba


def _transparent() -> np.ndarray:
    rgba = np.zeros((5, 7, 4), dtype=np.uint8); yy, xx = np.mgrid[:5, :7]; rgba[..., :3] = np.stack((xx * 31, yy * 47, (xx + yy) * 19), axis=2); return rgba


def _line_art() -> np.ndarray:
    rgba = np.zeros((9, 11, 4), dtype=np.uint8); rgba[..., :3] = (0, 240, 120); rgba[4, 1:10, 3] = 255; return rgba


def _odd() -> np.ndarray:
    rgba = np.zeros((5, 7, 4), dtype=np.uint8); rgba[..., :3] = (180, 60, 230); rgba[1:4, 2:5, 3] = 255; rgba[0, 3, 3] = 128; return rgba


CASES: tuple[tuple[str, Callable[[], np.ndarray]], ...] = (
    ("opaque_flat", _opaque_flat), ("hard_chroma_split", _hard_chroma_split),
    ("partial_alpha_colored_edge", _partial_edge), ("hidden_rgb_a", lambda: _hidden_rgb((0, 0, 0))),
    ("hidden_rgb_b", lambda: _hidden_rgb((255, 17, 241))), ("transparent_asset", _transparent),
    ("line_art_color", _line_art), ("odd_dimensions", _odd),
)


def _normal(array: np.ndarray) -> tuple[np.ndarray, str]:
    value = np.asarray(array)
    if value.dtype.kind == "f": return np.ascontiguousarray(value.astype("<f4")), "little-endian float32"
    if value.dtype.kind in "bu": return np.ascontiguousarray(value.astype(np.uint8)), "uint8"
    if value.dtype.itemsize <= 4: return np.ascontiguousarray(value.astype("<i4")), "little-endian int32"
    return np.ascontiguousarray(value.astype("<i8")), "little-endian int64"


def _fixed_arrays(rgba: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    alpha_lr = rgba[..., 3].astype(np.float32) / 255.0
    alpha_hr = sal_classic.resize_alpha_main_x4(alpha_lr)
    rgb_hr, diag = sal_classic.segment_guided_normalized_lanczos_rgb(rgba[..., :3], alpha_lr, alpha_hr, scale=SCALE, **main_sal_classic_rgb_options())
    final = np.dstack((rgb_hr, np.rint(alpha_hr * 255.0).astype(np.uint8)))
    if _sha(final.tobytes()) != EXPECTED_FIXED_RGBA_SHA256: raise ValueError("fixed Color-SDF output SHA-256 drifted")
    base = resize_premultiplied_rgb_lanczos(rgba[..., :3], alpha_lr, scale=SCALE)
    base_linear = _srgb_to_linear(base.astype(np.float32) / 255.0)
    final_linear = _srgb_to_linear(rgb_hr.astype(np.float32) / 255.0)
    graph = alpha_sdf._extract_segment_graph(alpha_lr, level=0.5)
    arrays: dict[str, np.ndarray] = {
        "fixed/input_rgba_u8": rgba, "fixed/alpha_lr_f32": alpha_lr, "fixed/alpha_hr_f32": alpha_hr,
        "fixed/premultiplied_lanczos_rgb_u8": base, "fixed/base_linear_rgb_f32": base_linear,
        "fixed/final_rgb_u8": rgb_hr, "fixed/final_rgba_u8": final, "fixed/final_linear_rgb_f32": final_linear,
        "fixed/gate_hr_f32": diag.gate_hr, "fixed/condition_hr_f32": diag.condition_hr,
        "fixed/contour_support_hr_f32": diag.contour_support_hr, "fixed/residual_luma_hr_f32": diag.residual_luma_hr,
        "fixed/active_mask_hr_u8": diag.active_mask_hr,
        "fixed/applied_residual_linear_f32": np.where(diag.gate_hr[..., None] > 0, (final_linear - base_linear) / np.maximum(diag.gate_hr[..., None], 1e-8), 0.0),
    }
    if graph is not None:
        arrays |= {"fixed/contour_starts_f32": graph.starts, "fixed/contour_ends_f32": graph.ends, "fixed/contour_tangents_f32": graph.tangents, "fixed/contour_ids_i32": graph.contour_ids, "fixed/contour_arc_midpoints_f32": graph.arc_midpoints, "fixed/contour_lengths_f32": graph.contour_lengths, "fixed/contour_closed_u8": graph.contour_closed}
    scalars = {"active_mask_sum": int(diag.active_mask_hr.sum()), "expected_final_rgba_sha256": EXPECTED_FIXED_RGBA_SHA256}
    return arrays, scalars


def _build() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    fixed = _fixture(); arrays, scalars = _fixed_arrays(fixed); case_entries = []
    for name, factory in CASES:
        rgba = factory(); output, diagnostics = upscale_color_sdf_x4(rgba)
        arrays[f"cases/{name}/input_rgba_u8"] = rgba; arrays[f"cases/{name}/output_rgba_u8"] = output
        case_entries.append({"id": name, "input_shape": list(rgba.shape), "output_shape": list(output.shape), "output_sha256": _sha(output.tobytes()), "active_mask_sum": int(diagnostics.active_mask_hr.sum())})
    entries = []; normalized = {}
    for name, array in sorted(arrays.items()):
        value, dtype = _normal(array); normalized[name] = value; raw = value.tobytes()
        entries.append({"id": name, "path": f"{name}.raw", "shape": list(value.shape), "dtype": dtype, "byte_length": len(raw), "sha256": _sha(raw)})
    manifest = {"schema_version": 1, "kind": "python_color_sdf_sal_classic_x4_parity", "pipeline_id": sal_classic.SAL_CLASSIC_PIPELINE_ID, "scale": SCALE, "options": main_sal_classic_rgb_options(), "versions": {item: importlib.metadata.version(item) for item in ("numpy", "pillow", "scipy", "scikit-image", "sdf-upscaler")}, "fixed_scalars": scalars, "absent_stages": {"raw_signed_weights": "local chunk arrays are not exposed by canonical SAL without source changes", "unclamped_candidate_rgb": "local chunk candidate is not returned by canonical SAL", "local_range_clamp_bounds": "local chunk bounds are not returned by canonical SAL"}, "arrays": entries, "cases": case_entries}
    return manifest, normalized


def generate(output: Path) -> None:
    manifest, arrays = _build(); output.mkdir(parents=True, exist_ok=True)
    for entry in manifest["arrays"]:
        path = output / entry["path"]; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(arrays[entry["id"]].tobytes())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8"); verify(output)


def verify(output: Path) -> None:
    stored = json.loads((output / "manifest.json").read_text(encoding="utf-8")); regenerated, arrays = _build()
    if stored != regenerated: raise ValueError("manifest does not match a fresh canonical run")
    for entry in stored["arrays"]:
        raw = (output / entry["path"]).read_bytes(); expected = arrays[entry["id"]].tobytes()
        if raw != expected or len(raw) != entry["byte_length"] or _sha(raw) != entry["sha256"]: raise ValueError(f"fixture mismatch: {entry['id']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, default=OUTPUT); parser.add_argument("--verify", action="store_true"); args = parser.parse_args()
    if args.verify: verify(args.output)
    else: generate(args.output)
    return 0


if __name__ == "__main__": raise SystemExit(main())
