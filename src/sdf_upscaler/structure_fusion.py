"""Output-gated RGB reconstruction from an edge-local geometric field.

This is the stage-1 experiment described in ``LLM.md``.  It is deliberately not
an SVG renderer: linked raster contours provide local segment frames, signed
distance, side colours, and confidence.  The resulting structure residual is
added to an existing HAT result and can be rejected by LR reprojection evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import (
    distance_transform_edt,
    map_coordinates,
    maximum_filter,
    minimum_filter,
    uniform_filter,
)

from .rgb_gate import _linear_to_srgb, _srgb_to_linear, resize_premultiplied_rgb_lanczos
from .structure_fields import StructureField, extract_structure_field


@dataclass(frozen=True)
class StructureFusionDiagnostics:
    """Dense evidence and residuals produced by output-gated structure fusion."""

    structure_candidate_hr: np.ndarray
    structure_residual_hr: np.ndarray
    structure_gate_hr: np.ndarray
    reprojection_gain_lr: np.ndarray
    reprojection_gate_hr: np.ndarray
    applied_gate_hr: np.ndarray
    unconstrained_rgb_hr: np.ndarray


def _as_rgb_u8(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError(f"{name} must be a uint8 HxWx3 array")
    return array


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    return (clipped * clipped * (3.0 - 2.0 * clipped)).astype(np.float32)


def _resize_plane(
    plane: np.ndarray,
    size: tuple[int, int],
    resample: Image.Resampling,
) -> np.ndarray:
    return np.asarray(
        Image.fromarray(np.asarray(plane, dtype=np.float32), mode="F").resize(size, resample),
        dtype=np.float32,
    )


def _resize_rgb_float(
    rgb: np.ndarray,
    size: tuple[int, int],
    resample: Image.Resampling,
) -> np.ndarray:
    return np.stack(
        [_resize_plane(rgb[..., channel], size, resample) for channel in range(3)],
        axis=2,
    ).astype(np.float32)


def _sample_rgb(rgb: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            map_coordinates(
                rgb[..., channel],
                (y, x),
                order=1,
                mode="nearest",
                prefilter=False,
            )
            for channel in range(3)
        ],
        axis=2,
    ).astype(np.float32)


def _downsample_rgb_linear(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return _resize_rgb_float(rgb, size, Image.Resampling.LANCZOS)


def edge_local_structure_fusion_x4(
    rgb_lr: np.ndarray,
    hat_rgb_hr: np.ndarray,
    *,
    structure_field: StructureField | None = None,
    scale: int = 4,
    propagation_radius: float = 6.0,
    side_offset_lr: float = 1.25,
    edge_softness_lr: float = 1.0,
    edge_band_start_lr: float = 0.25,
    edge_band_stop_lr: float = 1.75,
    contrast_start: float = 0.02,
    contrast_stop: float = 0.12,
    max_residual: float = 0.08,
    strength: float = 1.0,
    local_range_epsilon: float = 0.01,
    reprojection_gate_enabled: bool = True,
    reprojection_window: int = 5,
    reprojection_margin: float = 0.0,
    reprojection_transition: float = 0.001,
) -> tuple[np.ndarray, StructureFusionDiagnostics]:
    """Add an edge-local Segment-SDF residual to an existing HAT x4 result.

    Side colours are sampled around the nearest linked RGB contour in linear
    light.  Their SDF reconstruction is range-limited and converted into a
    residual relative to a linear-light Lanczos base.  Confidence and optional
    LR reprojection gates decide where that residual may modify HAT.
    """

    rgb = _as_rgb_u8(rgb_lr, "rgb_lr")
    hat = _as_rgb_u8(hat_rgb_hr, "hat_rgb_hr")
    if not isinstance(scale, int) or scale < 1:
        raise ValueError("scale must be a positive integer")
    expected_shape = (rgb.shape[0] * scale, rgb.shape[1] * scale, 3)
    if hat.shape != expected_shape:
        raise ValueError(f"hat_rgb_hr must have shape {expected_shape}, got {hat.shape}")
    if propagation_radius <= 0.0 or side_offset_lr <= 0.0 or edge_softness_lr <= 0.0:
        raise ValueError("geometry radii and edge softness must be positive")
    if not 0.0 <= edge_band_start_lr < edge_band_stop_lr:
        raise ValueError("edge band bounds must satisfy 0 <= start < stop")
    if not 0.0 <= contrast_start < contrast_stop <= 1.0:
        raise ValueError("contrast bounds must satisfy 0 <= start < stop <= 1")
    if max_residual < 0.0 or local_range_epsilon < 0.0:
        raise ValueError("residual and local-range bounds must be non-negative")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if reprojection_window < 1 or reprojection_margin < 0.0:
        raise ValueError("reprojection window must be positive and margin non-negative")
    if reprojection_transition <= 0.0:
        raise ValueError("reprojection transition must be positive")

    field = structure_field or extract_structure_field(
        rgb,
        propagation_radius=propagation_radius,
    )
    if field.features.shape[:2] != rgb.shape[:2]:
        raise ValueError("structure_field must match rgb_lr spatial size")

    output_size = (hat.shape[1], hat.shape[0])
    empty_hr = np.zeros(hat.shape[:2], dtype=np.float32)
    empty_rgb_hr = np.zeros_like(hat, dtype=np.float32)
    empty_lr = np.zeros(rgb.shape[:2], dtype=np.float32)
    if not field.edge_mask.any() or strength == 0.0 or max_residual == 0.0:
        diagnostics = StructureFusionDiagnostics(
            structure_candidate_hr=hat.astype(np.float32) / 255.0,
            structure_residual_hr=empty_rgb_hr,
            structure_gate_hr=empty_hr,
            reprojection_gain_lr=empty_lr,
            reprojection_gate_hr=empty_hr,
            applied_gate_hr=empty_hr,
            unconstrained_rgb_hr=hat.copy(),
        )
        return hat.copy(), diagnostics

    linear_lr = _srgb_to_linear(rgb.astype(np.float32) / 255.0)
    distance_lr, nearest = distance_transform_edt(~field.edge_mask, return_indices=True)
    nearest_y, nearest_x = nearest
    normal_x = field.channel("normal_x")
    normal_y = field.channel("normal_y")
    edge_normal_x = normal_x[nearest_y, nearest_x]
    edge_normal_y = normal_y[nearest_y, nearest_x]
    edge_y = nearest_y.astype(np.float32)
    edge_x = nearest_x.astype(np.float32)
    plus_lr = _sample_rgb(
        linear_lr,
        edge_y + edge_normal_y * side_offset_lr,
        edge_x + edge_normal_x * side_offset_lr,
    )
    minus_lr = _sample_rgb(
        linear_lr,
        edge_y - edge_normal_y * side_offset_lr,
        edge_x - edge_normal_x * side_offset_lr,
    )
    plus_hr = _resize_rgb_float(plus_lr, output_size, Image.Resampling.BILINEAR)
    minus_hr = _resize_rgb_float(minus_lr, output_size, Image.Resampling.BILINEAR)

    signed_distance_lr = field.channel("signed_distance") * propagation_radius
    signed_distance_hr = _resize_plane(
        signed_distance_lr,
        output_size,
        Image.Resampling.BICUBIC,
    )
    side_mix = _smoothstep(0.5 + signed_distance_hr / edge_softness_lr)
    candidate_linear = minus_hr + (plus_hr - minus_hr) * side_mix[..., None]

    local_min_lr = np.stack(
        [minimum_filter(linear_lr[..., channel], size=3, mode="nearest") for channel in range(3)],
        axis=2,
    )
    local_max_lr = np.stack(
        [maximum_filter(linear_lr[..., channel], size=3, mode="nearest") for channel in range(3)],
        axis=2,
    )
    local_min_hr = _resize_rgb_float(local_min_lr, output_size, Image.Resampling.BILINEAR)
    local_max_hr = _resize_rgb_float(local_max_lr, output_size, Image.Resampling.BILINEAR)
    candidate_linear = np.clip(
        candidate_linear,
        local_min_hr - local_range_epsilon,
        local_max_hr + local_range_epsilon,
    ).astype(np.float32)

    alpha_opaque = np.ones(rgb.shape[:2], dtype=np.float32)
    base_u8 = resize_premultiplied_rgb_lanczos(rgb, alpha_opaque, scale=scale)
    base_linear = _srgb_to_linear(base_u8.astype(np.float32) / 255.0)
    hat_linear = _srgb_to_linear(hat.astype(np.float32) / 255.0)
    residual = np.clip(candidate_linear - base_linear, -max_residual, max_residual)

    edge_confidence_hr = _resize_plane(
        field.channel("edge_confidence"),
        output_size,
        Image.Resampling.BILINEAR,
    )
    absolute_distance_hr = np.abs(signed_distance_hr)
    band_gate = 1.0 - _smoothstep(
        (absolute_distance_hr - edge_band_start_lr) / (edge_band_stop_lr - edge_band_start_lr)
    )
    contrast_lr = np.max(np.abs(plus_lr - minus_lr), axis=2)
    contrast_gate_lr = _smoothstep(
        (contrast_lr - contrast_start) / (contrast_stop - contrast_start)
    )
    contrast_gate_hr = _resize_plane(
        contrast_gate_lr,
        output_size,
        Image.Resampling.BILINEAR,
    )
    structure_gate = np.clip(
        edge_confidence_hr * band_gate * contrast_gate_hr,
        0.0,
        1.0,
    ).astype(np.float32)

    unconstrained_linear = np.clip(
        hat_linear + float(strength) * structure_gate[..., None] * residual,
        0.0,
        1.0,
    )
    unconstrained_u8 = np.rint(
        np.clip(_linear_to_srgb(unconstrained_linear), 0.0, 1.0) * 255.0
    ).astype(np.uint8)

    if reprojection_gate_enabled:
        lr_size = (rgb.shape[1], rgb.shape[0])
        hat_reprojected = _downsample_rgb_linear(hat_linear, lr_size)
        trial_reprojected = _downsample_rgb_linear(unconstrained_linear, lr_size)
        hat_error = uniform_filter(
            np.mean(np.abs(hat_reprojected - linear_lr), axis=2),
            size=reprojection_window,
            mode="nearest",
        )
        trial_error = uniform_filter(
            np.mean(np.abs(trial_reprojected - linear_lr), axis=2),
            size=reprojection_window,
            mode="nearest",
        )
        reprojection_gain = (hat_error - trial_error).astype(np.float32)
        reprojection_gate_lr = _smoothstep(
            (reprojection_gain - reprojection_margin) / reprojection_transition
        )
        reprojection_gate_hr = _resize_plane(
            reprojection_gate_lr,
            output_size,
            Image.Resampling.BILINEAR,
        )
    else:
        reprojection_gain = empty_lr
        reprojection_gate_hr = np.ones_like(empty_hr)

    applied_gate = (structure_gate * reprojection_gate_hr).astype(np.float32)
    output_linear = np.clip(
        hat_linear + float(strength) * applied_gate[..., None] * residual,
        0.0,
        1.0,
    )
    output_u8 = np.rint(np.clip(_linear_to_srgb(output_linear), 0.0, 1.0) * 255.0).astype(np.uint8)
    diagnostics = StructureFusionDiagnostics(
        structure_candidate_hr=np.clip(_linear_to_srgb(candidate_linear), 0.0, 1.0).astype(
            np.float32
        ),
        structure_residual_hr=residual.astype(np.float32),
        structure_gate_hr=structure_gate,
        reprojection_gain_lr=reprojection_gain,
        reprojection_gate_hr=reprojection_gate_hr.astype(np.float32),
        applied_gate_hr=applied_gate,
        unconstrained_rgb_hr=unconstrained_u8,
    )
    return output_u8, diagnostics
