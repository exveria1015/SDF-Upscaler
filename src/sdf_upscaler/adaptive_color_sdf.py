"""HAT-free adaptive RGB reconstruction for structured RGBA assets.

The alpha path has a scalar inside/outside field, so a single calibrated SDF
is sufficient.  RGB needs shared geometry plus vector-valued colours on both
sides of each interface.  This module reuses the existing eleven-channel RGB
structure field to build that interface candidate, then adds two conservative
steps that are useful for flat/vector-like art:

* a capped linear-light median residual in opaque regions;
* iterative premultiplied-RGB back-projection to the observed LR image.

SAL-Classic remains the compositing-correct base and owns partial-alpha colour.
The adaptive path is only allowed to modify opaque interior pixels.  A final
local reprojection gate and a whole-image fallback prevent the new candidate
from worsening LR premultiplied-RGB consistency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import (
    distance_transform_edt,
    maximum_filter,
    median_filter,
    minimum_filter,
    uniform_filter,
)

from .rgb_gate import (
    _linear_to_srgb,
    _resize_plane,
    _srgb_to_linear,
    resize_bounded_rgb_lanczos,
)
from .sal_classic import (
    SegmentGuidedRGBDiagnostics,
    main_sal_classic_rgb_options,
    segment_guided_normalized_lanczos_rgb,
)
from .structure_fields import extract_structure_field
from .structure_fusion import StructureFusionDiagnostics, edge_local_structure_fusion_x4

ADAPTIVE_COLOR_SDF_PIPELINE_ID = (
    "adaptive_color_sdf_rgb_x4_sal_base_structure_median_premultiplied_backprojection_v1"
)


@dataclass(frozen=True)
class AdaptiveColorSDFDiagnostics:
    """Evidence and guards emitted by :func:`adaptive_color_sdf_rgb_x4`."""

    sal: SegmentGuidedRGBDiagnostics
    structure: StructureFusionDiagnostics
    structure_interior_gate_hr: np.ndarray
    median_active_mask_hr: np.ndarray
    median_residual_luma_hr: np.ndarray
    reprojection_gate_lr: np.ndarray
    reprojection_gate_hr: np.ndarray
    lr_error_base: float
    lr_error_candidate_before: float
    lr_error_candidate_after: float
    lr_error_final: float
    global_reprojection_fallback: bool


def main_adaptive_color_sdf_options() -> dict[str, float | int]:
    """Return the conservative first public profile.

    Residual bounds are in linear RGB.  Distances are HR pixels except for the
    hidden-RGB extension, which is measured in LR pixels.
    """

    return {
        "hidden_rgb_radius": 16.0,
        "hidden_rgb_feather": 8.0,
        "opaque_threshold": 0.999,
        "interior_start_hr": 1.0,
        "interior_stop_hr": 4.0,
        "structure_max_residual": 0.02,
        "structure_strength": 1.0,
        "median_max_residual": 0.02,
        "backprojection_iterations": 2,
        "backprojection_strength": 0.5,
        "local_range_epsilon": 0.01,
        "reprojection_window": 5,
        "reprojection_transition": 0.0005,
    }


def _as_rgba_u8(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 4 or array.dtype != np.uint8:
        raise ValueError("rgba_lr must be a uint8 HxWx4 array")
    if array.shape[0] < 3 or array.shape[1] < 3:
        raise ValueError("rgba_lr height and width must both be at least three pixels")
    return array


def _as_alpha_hr(value: np.ndarray, expected: tuple[int, int]) -> np.ndarray:
    alpha = np.asarray(value, dtype=np.float32)
    if alpha.shape != expected:
        raise ValueError(f"alpha_hr must have shape {expected}, got {alpha.shape}")
    if not np.isfinite(alpha).all():
        raise ValueError("alpha_hr contains NaN or infinity")
    return np.clip(alpha, 0.0, 1.0)


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    return (clipped * clipped * (3.0 - 2.0 * clipped)).astype(np.float32)


def _bounded_visible_rgb(
    rgba: np.ndarray,
    *,
    radius: float,
    feather: float,
) -> np.ndarray:
    """Build a hidden-RGB-invariant foreground extension.

    Source RGB under exact alpha zero is never read.  This is stricter than the
    production HAT staging policy because a deterministic RGB reconstruction
    should be exactly invariant to arbitrary hidden colours.
    """

    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("hidden_rgb_radius must be finite and non-negative")
    if not np.isfinite(feather) or feather < 0.0 or feather > radius:
        raise ValueError("hidden_rgb_feather must be in [0, hidden_rgb_radius]")
    source_rgb = rgba[..., :3]
    known = rgba[..., 3] > 0
    result = np.zeros_like(source_rgb)
    result[known] = source_rgb[known]
    if radius == 0.0 or not known.any() or known.all():
        return result
    distance, indices = distance_transform_edt(~known, return_indices=True)
    if feather == 0.0:
        weight = (distance <= radius).astype(np.float32)
    else:
        weight = np.clip((radius - distance) / feather, 0.0, 1.0).astype(np.float32)
        weight = _smoothstep(weight)
    transparent = ~known
    nearest = source_rgb[indices[0], indices[1]].astype(np.float32)
    result[transparent] = np.rint(nearest[transparent] * weight[transparent, None]).astype(np.uint8)
    return result


def _resize_rgb_float(
    rgb: np.ndarray,
    size: tuple[int, int],
    resample: Image.Resampling,
) -> np.ndarray:
    return np.stack(
        [_resize_plane(rgb[..., channel], size, resample) for channel in range(3)],
        axis=2,
    ).astype(np.float32)


def _downsample_premultiplied(
    straight_linear_hr: np.ndarray,
    alpha_hr: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    premultiplied = straight_linear_hr * alpha_hr[..., None]
    return _resize_rgb_float(premultiplied, size, Image.Resampling.LANCZOS)


def _validate_options(options: dict[str, float | int]) -> None:
    if not 0.0 < float(options["opaque_threshold"]) <= 1.0:
        raise ValueError("opaque_threshold must be in (0, 1]")
    if not 0.0 <= float(options["interior_start_hr"]) < float(options["interior_stop_hr"]):
        raise ValueError("interior bounds require 0 <= start < stop")
    for name in (
        "structure_max_residual",
        "structure_strength",
        "median_max_residual",
        "backprojection_strength",
        "local_range_epsilon",
        "reprojection_transition",
    ):
        if not np.isfinite(float(options[name])) or float(options[name]) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if float(options["structure_strength"]) > 1.0:
        raise ValueError("structure_strength must be at most one")
    if float(options["backprojection_strength"]) > 1.0:
        raise ValueError("backprojection_strength must be at most one")
    if int(options["backprojection_iterations"]) < 0:
        raise ValueError("backprojection_iterations must be non-negative")
    if int(options["reprojection_window"]) < 1:
        raise ValueError("reprojection_window must be positive")


def adaptive_color_sdf_rgb_x4(
    rgba_lr: np.ndarray,
    alpha_hr: np.ndarray,
    *,
    options: dict[str, float | int] | None = None,
    sal_rgb_options: dict[str, object] | None = None,
) -> tuple[np.ndarray, AdaptiveColorSDFDiagnostics]:
    """Reconstruct straight RGB x4 while preserving the supplied alpha.

    The supplied ``alpha_hr`` may come from deterministic SAL, native alpha, or
    a learned M-alpha checkpoint.  The function never changes it.
    """

    rgba = _as_rgba_u8(rgba_lr)
    expected_shape = (rgba.shape[0] * 4, rgba.shape[1] * 4)
    alpha_high = _as_alpha_hr(alpha_hr, expected_shape)
    effective = main_adaptive_color_sdf_options()
    if options is not None:
        unknown = set(options) - set(effective)
        if unknown:
            raise ValueError(f"unknown Adaptive Color-SDF options: {sorted(unknown)}")
        effective.update(options)
    _validate_options(effective)

    alpha_low = rgba[..., 3].astype(np.float32) / 255.0
    sal_options = main_sal_classic_rgb_options()
    if sal_rgb_options is not None:
        unknown = set(sal_rgb_options) - set(sal_options)
        if unknown:
            raise ValueError(f"unknown SAL-Classic RGB options: {sorted(unknown)}")
        sal_options.update(sal_rgb_options)
    base_u8, sal_diagnostics = segment_guided_normalized_lanczos_rgb(
        rgba[..., :3],
        alpha_low,
        alpha_high,
        scale=4,
        **sal_options,
    )
    base_linear = _srgb_to_linear(base_u8.astype(np.float32) / 255.0)

    bounded = _bounded_visible_rgb(
        rgba,
        radius=float(effective["hidden_rgb_radius"]),
        feather=float(effective["hidden_rgb_feather"]),
    )
    bounded_base_u8 = resize_bounded_rgb_lanczos(bounded, scale=4)
    bounded_base_linear = _srgb_to_linear(bounded_base_u8.astype(np.float32) / 255.0)
    structure_field = extract_structure_field(bounded)
    structure_u8, structure_diagnostics = edge_local_structure_fusion_x4(
        bounded,
        bounded_base_u8,
        structure_field=structure_field,
        scale=4,
        max_residual=float(effective["structure_max_residual"]),
        strength=1.0,
        reprojection_gate_enabled=False,
    )
    structure_linear = _srgb_to_linear(structure_u8.astype(np.float32) / 255.0)
    structure_residual = structure_linear - bounded_base_linear

    opaque = alpha_high >= float(effective["opaque_threshold"])
    interior_distance = distance_transform_edt(opaque).astype(np.float32)
    interior_gate = _smoothstep(
        (interior_distance - float(effective["interior_start_hr"]))
        / (float(effective["interior_stop_hr"]) - float(effective["interior_start_hr"]))
    )
    candidate = np.clip(
        base_linear
        + (float(effective["structure_strength"]) * interior_gate[..., None] * structure_residual),
        0.0,
        1.0,
    )

    median = median_filter(candidate, size=(3, 3, 1), mode="nearest")
    median_residual = np.clip(
        median - candidate,
        -float(effective["median_max_residual"]),
        float(effective["median_max_residual"]),
    )
    candidate[opaque] = np.clip(candidate[opaque] + median_residual[opaque], 0.0, 1.0)
    median_luma = np.sum(
        np.abs(median_residual) * np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32),
        axis=2,
    )
    median_active = opaque & (median_luma > 1e-8)

    # Use the same hidden-RGB-invariant field for both local colour bounds and
    # premultiplied reprojection. Multiplication by alpha makes the extended
    # samples exactly irrelevant to the LR target while preventing arbitrary
    # alpha-zero source colours from changing the clamp.
    lr_linear = _srgb_to_linear(bounded.astype(np.float32) / 255.0)
    lr_premultiplied = lr_linear * alpha_low[..., None]
    lr_size = (rgba.shape[1], rgba.shape[0])
    base_reprojected = _downsample_premultiplied(base_linear, alpha_high, lr_size)
    candidate_before = _downsample_premultiplied(candidate, alpha_high, lr_size)
    base_error_map = np.mean(np.abs(base_reprojected - lr_premultiplied), axis=2)
    candidate_before_error = float(np.mean(np.abs(candidate_before - lr_premultiplied)))

    local_min_lr = np.stack(
        [minimum_filter(lr_linear[..., channel], size=3, mode="nearest") for channel in range(3)],
        axis=2,
    )
    local_max_lr = np.stack(
        [maximum_filter(lr_linear[..., channel], size=3, mode="nearest") for channel in range(3)],
        axis=2,
    )
    output_size = (expected_shape[1], expected_shape[0])
    local_min_hr = _resize_rgb_float(local_min_lr, output_size, Image.Resampling.BILINEAR)
    local_max_hr = _resize_rgb_float(local_max_lr, output_size, Image.Resampling.BILINEAR)
    range_epsilon = float(effective["local_range_epsilon"])
    backprojected = candidate.copy()
    safe_alpha = np.maximum(alpha_high, 1e-6)
    for _ in range(int(effective["backprojection_iterations"])):
        reprojected = _downsample_premultiplied(backprojected, alpha_high, lr_size)
        error_lr = lr_premultiplied - reprojected
        error_hr = _resize_rgb_float(error_lr, output_size, Image.Resampling.LANCZOS)
        correction = error_hr / safe_alpha[..., None]
        backprojected[opaque] += float(effective["backprojection_strength"]) * correction[opaque]
        backprojected[opaque] = np.clip(
            backprojected[opaque],
            (local_min_hr - range_epsilon)[opaque],
            (local_max_hr + range_epsilon)[opaque],
        )
        backprojected = np.clip(backprojected, 0.0, 1.0)

    candidate_after = _downsample_premultiplied(backprojected, alpha_high, lr_size)
    candidate_error_map = np.mean(np.abs(candidate_after - lr_premultiplied), axis=2)
    base_error_smooth = uniform_filter(
        base_error_map,
        size=int(effective["reprojection_window"]),
        mode="nearest",
    )
    candidate_error_smooth = uniform_filter(
        candidate_error_map,
        size=int(effective["reprojection_window"]),
        mode="nearest",
    )
    gain = base_error_smooth - candidate_error_smooth
    transition = max(float(effective["reprojection_transition"]), 1e-8)
    reprojection_gate_lr = _smoothstep(gain / transition)
    reprojection_gate_hr = _resize_plane(
        reprojection_gate_lr,
        output_size,
        Image.Resampling.BILINEAR,
    )
    guarded = np.clip(
        base_linear + reprojection_gate_hr[..., None] * (backprojected - base_linear),
        0.0,
        1.0,
    )
    guarded[~opaque] = base_linear[~opaque]
    guarded_reprojected = _downsample_premultiplied(guarded, alpha_high, lr_size)
    base_error = float(np.mean(np.abs(base_reprojected - lr_premultiplied)))
    candidate_after_error = float(np.mean(np.abs(candidate_after - lr_premultiplied)))
    guarded_error = float(np.mean(np.abs(guarded_reprojected - lr_premultiplied)))
    global_fallback = guarded_error > base_error + 1e-9
    final_linear = base_linear if global_fallback else guarded
    final_error = base_error if global_fallback else guarded_error

    final_linear[alpha_high <= 1e-6] = 0.0
    output_u8 = np.rint(np.clip(_linear_to_srgb(final_linear), 0.0, 1.0) * 255.0).astype(np.uint8)
    diagnostics = AdaptiveColorSDFDiagnostics(
        sal=sal_diagnostics,
        structure=structure_diagnostics,
        structure_interior_gate_hr=interior_gate,
        median_active_mask_hr=median_active,
        median_residual_luma_hr=median_luma,
        reprojection_gate_lr=reprojection_gate_lr,
        reprojection_gate_hr=reprojection_gate_hr.astype(np.float32),
        lr_error_base=base_error,
        lr_error_candidate_before=candidate_before_error,
        lr_error_candidate_after=candidate_after_error,
        lr_error_final=final_error,
        global_reprojection_fallback=global_fallback,
    )
    return output_u8, diagnostics


def upscale_rgba_adaptive_color_sdf_x4(
    rgba_lr: np.ndarray,
    *,
    alpha_hr: np.ndarray | None = None,
    options: dict[str, float | int] | None = None,
    sal_rgb_options: dict[str, object] | None = None,
) -> tuple[np.ndarray, AdaptiveColorSDFDiagnostics]:
    """Upscale RGBA x4 with adaptive RGB and supplied or deterministic alpha."""

    from .alpha_sdf import resize_alpha_main_x4

    rgba = _as_rgba_u8(rgba_lr)
    alpha_low = rgba[..., 3].astype(np.float32) / 255.0
    selected_alpha = resize_alpha_main_x4(alpha_low) if alpha_hr is None else alpha_hr
    selected_alpha = _as_alpha_hr(
        selected_alpha,
        (rgba.shape[0] * 4, rgba.shape[1] * 4),
    )
    rgb_hr, diagnostics = adaptive_color_sdf_rgb_x4(
        rgba,
        selected_alpha,
        options=options,
        sal_rgb_options=sal_rgb_options,
    )
    alpha_u8 = np.rint(selected_alpha * 255.0).astype(np.uint8)
    return np.dstack((rgb_hr, alpha_u8)), diagnostics
