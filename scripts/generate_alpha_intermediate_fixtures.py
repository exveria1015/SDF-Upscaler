"""Freeze canonical Python Alpha-SDF intermediate and primitive parity fixtures.

Run with ``uv run python scripts/generate_alpha_intermediate_fixtures.py``.
``--verify`` re-runs the canonical implementation and compares every declared
little-endian raw array, shape, dtype, byte length, and SHA-256 digest.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import convolve, sobel, uniform_filter
from scipy.spatial import cKDTree
from skimage.measure import find_contours

from sdf_upscaler import alpha_sdf as canonical
from sdf_upscaler import main_alpha_x4_options

SCHEMA_VERSION = 1
SCALE = 4
INPUT = Path("fixtures/parity/alpha/input_alpha_u8.raw")
INPUT_MANIFEST = Path("fixtures/parity/alpha/manifest.json")
OUTPUT = Path("fixtures/parity/alpha_intermediates")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _float32(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(array, dtype="<f4"))


def _int32(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(array, dtype="<i4"))


def _int64(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(array, dtype="<i8"))


def _u8(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(array, dtype=np.uint8))


def _input_alpha() -> np.ndarray:
    source = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
    width, height = source["input"]["width"], source["input"]["height"]
    raw = INPUT.read_bytes()
    if _sha256(raw) != source["input"]["sha256"] or len(raw) != width * height:
        raise ValueError("the fixed alpha fixture does not match its manifest")
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width).astype(np.float32) / 255.0


def _stage_arrays(alpha: np.ndarray) -> dict[str, np.ndarray]:
    options = main_alpha_x4_options()
    geometry_keys = set(inspect.signature(canonical.alpha_geometry_candidate).parameters)
    geometry_options = {key: value for key, value in options.items() if key in geometry_keys}
    base, candidate, reliability = canonical.alpha_geometry_candidate(
        alpha, scale=SCALE, **geometry_options
    )
    ungated = canonical.apply_bounded_alpha_residual(
        base, candidate, reliability,
        max_residual=options["max_residual"], strength=options["strength"],
        gate_floor=options["gate_floor"], edge_gate=options["residual_edge_gate"],
        topology_lock=options["topology_lock"],
    )
    gate, diagnostic = canonical.broad_soft_reprojection_gate(
        alpha, base, ungated, scale=SCALE,
        broad_soft_window=options["broad_soft_window"],
        broad_soft_start=options["broad_soft_start"], broad_soft_stop=options["broad_soft_stop"],
        reprojection_window=options["reprojection_window"],
        reprojection_margin=options["reprojection_margin"],
        reprojection_transition=options["reprojection_transition"],
        support_gate_enabled=options["support_gate_enabled"], support_window=options["support_window"],
        support_start=options["support_start"], support_stop=options["support_stop"],
        support_centerline_bypass_enabled=options["support_centerline_bypass_enabled"],
        support_centerline_radius_limit=options["support_centerline_radius_limit"],
        support_centerline_min_component=options["support_centerline_min_component"],
        support_centerline_neighborhood=options["support_centerline_neighborhood"],
        support_centerline_peak_min=options["support_centerline_peak_min"],
        support_centerline_peak_stop=options["support_centerline_peak_stop"],
        support_centerline_max_foreground_area=options["support_centerline_max_foreground_area"],
        support_centerline_ratio_start=options["support_centerline_ratio_start"],
        support_centerline_ratio_stop=options["support_centerline_ratio_stop"],
    )
    protected = canonical.apply_bounded_alpha_residual(
        base, candidate, reliability,
        max_residual=options["max_residual"], strength=options["strength"],
        gate_floor=options["gate_floor"], edge_gate=options["residual_edge_gate"],
        topology_lock=options["topology_lock"], additional_gate=gate,
    )
    final, scs_lift = canonical.apply_conditional_scs_lift(
        alpha, protected, scale=SCALE,
        medial_radius_limit=options["scs_medial_radius_limit"],
        centerline_gate_threshold=options["scs_centerline_gate_threshold"],
        min_centerline_component=options["scs_min_centerline_component"],
        max_foreground_area=options["scs_max_foreground_area"], max_lift=options["scs_max_lift"],
        require_hard_foreground=options["scs_require_hard_foreground"],
    )
    graph = canonical._extract_segment_graph(alpha, level=0.5)
    if graph is None:
        raise ValueError("fixed alpha fixture unexpectedly has no contour graph")
    lr_y, lr_x = np.mgrid[:alpha.shape[0], :alpha.shape[1]]
    lr_query = np.stack((lr_y.ravel(), lr_x.ravel()), axis=1).astype(np.float32)
    lr_indices, _, lr_distance = canonical._query_segment_graph(graph, lr_query, nearest_segments=12)
    lr_tangents = graph.tangents[lr_indices]
    lr_normals = np.stack((-lr_tangents[:, 1], lr_tangents[:, 0]), axis=1)
    lr_sign = np.where(alpha.ravel() >= 0.5, 1.0, -1.0).astype(np.float32)
    analytic_lr = canonical._planar_segment_coverage(
        lr_sign * lr_distance, lr_normals, pixel_size_lr=1.0
    ).reshape(alpha.shape)
    hr_query = canonical._hr_queries(base.shape, SCALE)
    hr_indices, _, hr_distance = canonical._query_segment_graph(graph, hr_query, nearest_segments=12)
    hr_tangents = graph.tangents[hr_indices]
    hr_normals = np.stack((-hr_tangents[:, 1], hr_tangents[:, 0]), axis=1)
    hr_sign = np.where(base.ravel() >= 0.5, 1.0, -1.0).astype(np.float32)
    segment_hr = canonical._planar_segment_coverage(
        hr_sign * hr_distance, hr_normals, pixel_size_lr=1.0 / SCALE
    ).reshape(base.shape)
    intermediate_mass = 4.0 * alpha * (1.0 - alpha)
    local_kernel = np.full((5, 5), 1.0 / 25.0, dtype=np.float32)
    base_reprojected = np.asarray(
        Image.fromarray(base, mode="F").resize((alpha.shape[1], alpha.shape[0]), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    candidate_reprojected = np.asarray(
        Image.fromarray(ungated, mode="F").resize((alpha.shape[1], alpha.shape[0]), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    arrays: dict[str, np.ndarray] = {
        "stages/base_lanczos_hr_f32": base,
        "stages/candidate_hr_f32": candidate,
        "stages/ungated_hr_f32": ungated,
        "stages/protected_hr_f32": protected,
        "stages/final_hr_f32": final,
        "stages/scs_lift_hr_f32": scs_lift,
        "stages/reliability_hr_f32": reliability,
        "stages/reliable_sample_mask_hr_u8": _u8(reliability > 1e-4),
        "calibration/combined_gate_hr_f32": gate,
        "calibration/intermediate_mass_lr_f32": intermediate_mass,
        "calibration/base_reprojected_lr_f32": base_reprojected,
        "calibration/candidate_reprojected_lr_f32": candidate_reprojected,
        "geometry/analytic_coverage_lr_f32": analytic_lr,
        "geometry/residual_lr_f32": alpha - analytic_lr,
        "geometry/segment_coverage_hr_f32": segment_hr,
        "geometry/lr_nearest_segment_i64": _int64(lr_indices),
        "geometry/hr_nearest_segment_i64": _int64(hr_indices.reshape(base.shape)),
        "geometry/contour_starts_f32": graph.starts,
        "geometry/contour_ends_f32": graph.ends,
        "geometry/contour_tangents_f32": graph.tangents,
        "geometry/contour_ids_i32": _int32(graph.contour_ids),
        "geometry/contour_arc_midpoints_f32": graph.arc_midpoints,
        "geometry/contour_lengths_f32": graph.contour_lengths,
        "geometry/contour_closed_u8": _u8(graph.contour_closed),
        "geometry/segment_order_i32": _int32(np.arange(len(graph.starts))),
        "filters/sobel_x_lr_f32": sobel(alpha, axis=1, mode="nearest") / 8.0,
        "filters/sobel_y_lr_f32": sobel(alpha, axis=0, mode="nearest") / 8.0,
        "filters/local_background_lr_f32": convolve((alpha <= 0.05).astype(np.float32), local_kernel, mode="nearest"),
        "filters/local_foreground_lr_f32": convolve((alpha >= 0.95).astype(np.float32), local_kernel, mode="nearest"),
        "filters/broad_soft_uniform_lr_f32": uniform_filter(intermediate_mass.astype(np.float32), size=5, mode="nearest"),
    }
    for name in diagnostic.__dataclass_fields__:
        arrays[f"calibration/{name}_f32"] = getattr(diagnostic, name)
    return arrays


def _primitive_arrays() -> dict[str, np.ndarray]:
    source = np.asarray([[0.0, 0.25, 1.0], [0.5, 0.75, 0.125]], dtype=np.float32)
    filter_source = np.asarray([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]], dtype=np.float32)
    kernel = np.asarray([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32) / 6.0
    ambiguous = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    border = np.asarray([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    starts = np.stack((np.zeros(12), np.arange(12, dtype=np.float32)), axis=1)
    ends = starts + np.asarray([0.0, 1.0], dtype=np.float32)
    tangents = np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (12, 1))
    graph = canonical._SegmentGraph(
        starts=starts, ends=ends, tangents=tangents, contour_ids=np.zeros(12, dtype=np.int32),
        arc_midpoints=np.arange(12, dtype=np.float32), contour_lengths=np.asarray([12.0], dtype=np.float32),
        contour_closed=np.asarray([False]), tree=cKDTree((starts + ends) * 0.5),
    )
    tie_points = np.asarray([[0.0, 6.0], [0.0, 6.5], [0.0, 5.5]], dtype=np.float32)
    tree_distances, tree_indices = graph.tree.query(tie_points, k=12, workers=-1)
    exact_indices, exact_closest, exact_distances = canonical._query_segment_graph(graph, tie_points, nearest_segments=12)
    ambiguous_contours = find_contours(
        ambiguous, 0.5, fully_connected="high", positive_orientation="high"
    )
    border_contours = find_contours(border, 0.5, fully_connected="high", positive_orientation="high")
    arrays = {
        "pillow/source_f32": source,
        "pillow/bilinear_7x5_f32": np.asarray(Image.fromarray(source, mode="F").resize((7, 5), Image.Resampling.BILINEAR)),
        "pillow/lanczos_7x5_f32": np.asarray(Image.fromarray(source, mode="F").resize((7, 5), Image.Resampling.LANCZOS)),
        "contours/ambiguous_input_f32": ambiguous,
        "contours/ambiguous_points_f32": np.concatenate(ambiguous_contours),
        "contours/ambiguous_lengths_i32": _int32([len(item) for item in ambiguous_contours]),
        "contours/border_input_f32": border,
        "contours/border_points_f32": np.concatenate(border_contours),
        "contours/border_lengths_i32": _int32([len(item) for item in border_contours]),
        "scipy/source_f32": filter_source,
        "scipy/sobel_x_nearest_f32": sobel(filter_source, axis=1, mode="nearest"),
        "scipy/sobel_y_nearest_f32": sobel(filter_source, axis=0, mode="nearest"),
        "scipy/convolve_nearest_f32": convolve(filter_source, kernel, mode="nearest"),
        "scipy/uniform_3_nearest_f32": uniform_filter(filter_source, size=3, mode="nearest"),
        "k12/query_points_f32": tie_points,
        "k12/tree_distances_f32": tree_distances,
        "k12/tree_indices_i64": _int64(tree_indices),
        "k12/exact_indices_i64": _int64(exact_indices),
        "k12/exact_closest_f32": exact_closest,
        "k12/exact_distances_f32": exact_distances,
    }
    return arrays


def _canonical_bytes(array: np.ndarray) -> tuple[np.ndarray, str]:
    value = np.asarray(array)
    if value.dtype.kind == "f":
        return _float32(value), "little-endian float32"
    if value.dtype.kind == "i":
        return (_int64(value) if value.dtype.itemsize > 4 else _int32(value)), f"little-endian int{value.dtype.itemsize * 8}"
    if value.dtype.kind in "bu":
        return _u8(value), "uint8"
    raise ValueError(f"unsupported fixture dtype: {value.dtype}")


def _build() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    arrays = _stage_arrays(_input_alpha()) | _primitive_arrays()
    entries = []
    normalized = {}
    for name, array in sorted(arrays.items()):
        value, dtype = _canonical_bytes(array)
        normalized[name] = value
        raw = value.tobytes()
        entries.append({"id": name, "path": f"{name}.raw", "shape": list(value.shape), "dtype": dtype, "byte_length": len(raw), "sha256": _sha256(raw)})
    manifest = {
        "schema_version": SCHEMA_VERSION, "kind": "python_alpha_sdf_intermediate_parity",
        "input_fixture": {"path": str(INPUT), "manifest": str(INPUT_MANIFEST), "sha256": _sha256(INPUT.read_bytes())},
        "pipeline": {"scale": SCALE, "options": main_alpha_x4_options()},
        "versions": {package: importlib.metadata.version(package) for package in ("numpy", "pillow", "scipy", "scikit-image", "sdf-upscaler")},
        "absent_stages": [], "arrays": entries,
    }
    return manifest, normalized


def generate(output: Path) -> None:
    manifest, arrays = _build()
    output.mkdir(parents=True, exist_ok=True)
    for entry in manifest["arrays"]:
        path = output / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(arrays[entry["id"]].tobytes())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    verify(output)


def verify(output: Path) -> None:
    stored = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    regenerated, arrays = _build()
    if stored != regenerated:
        raise ValueError("manifest does not match a fresh canonical Python run")
    for entry in stored["arrays"]:
        actual = (output / entry["path"]).read_bytes()
        expected = arrays[entry["id"]].tobytes()
        if actual != expected or len(actual) != entry["byte_length"] or _sha256(actual) != entry["sha256"]:
            raise ValueError(f"fixture mismatch: {entry['id']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.output)
    else:
        generate(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
