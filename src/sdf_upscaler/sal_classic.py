"""HAT-free SAL-Classic x4 RGBA upscaling.

The alpha path is the calibrated Segment-SDF / adaptive-Lanczos route from
``alpha_sdf``.  RGB starts from a compositing-correct linear-light,
premultiplied Lanczos resize.  Only the alpha boundary band is refined with a
contour-aware normalized Lanczos candidate, applied as a bounded residual.

This module is intentionally deterministic.  It does not hallucinate texture
inside opaque regions; those regions remain the linear-light Lanczos baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve, sobel

from .alpha_sdf import (
    _bilinear_sample,
    _extract_segment_graph,
    _hr_queries,
    _lanczos_weight,
    _query_segment_frame,
    _query_segment_graph,
    persistent_centerline_support_gate,
    resize_alpha_main_x4,
)
from .rgb_gate import (
    _linear_to_srgb,
    _srgb_to_linear,
    resize_premultiplied_rgb_lanczos,
)

SAL_CLASSIC_PIPELINE_ID = (
    "sal_classic_x4_alpha_sal_r0_gsupport_conditional_scs_"
    "rgb_linear_premultiplied_lanczos_segment_guided_residual_v1"
)


@dataclass(frozen=True)
class SegmentGuidedRGBDiagnostics:
    """HR diagnostics for the segment-guided RGB residual."""

    gate_hr: np.ndarray
    condition_hr: np.ndarray
    contour_support_hr: np.ndarray
    residual_luma_hr: np.ndarray
    active_mask_hr: np.ndarray


def _as_rgb_u8(rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("rgb_lr must have shape HxWx3")
    if array.dtype != np.uint8:
        raise ValueError("rgb_lr must use uint8 values")
    return array


def _as_alpha(alpha: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(alpha, dtype=np.float32)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return np.clip(array, 0.0, 1.0)


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    return (clipped * clipped * (3.0 - 2.0 * clipped)).astype(np.float32)


def main_sal_classic_rgb_options() -> dict[str, object]:
    """Return the conservative x4 RGB profile used by SAL-Classic.

    The initial public profile keeps the Lanczos kernel isotropic.  Segment
    guidance is still active through contour provenance, arc-distance support,
    alpha normalization, hard-edge reliability, and the final residual gate.
    An anisotropic tangent/normal kernel remains available as an ablation.
    """

    return {
        "level": 0.5,
        "nearest_segments": 12,
        "support_radius_normal": 3.0,
        "support_radius_tangent": 3.0,
        "lanczos_lobes": 3,
        "steer_kernel": False,
        "edge_band_hr": 3.5,
        "condition_limit": 8.0,
        "condition_soft_start": 2.0,
        "arc_radius": 6.0,
        "contour_gate_enabled": True,
        "alpha_sample_floor": 1.0 / 255.0,
        "alpha_mass_floor": 0.03,
        "local_range_epsilon": 0.01,
        "max_residual": 0.20,
        "strength": 1.0,
        "hard_edge_gate": True,
        "line_art_bypass_enabled": True,
        "line_art_radius_limit": 0.75,
        "line_art_min_component": 3,
        "line_art_neighborhood": 3,
        "line_art_peak_min": 0.8,
        "line_art_max_foreground_area": 4096,
        "line_art_max_canvas_area": 16384,
        "line_art_ratio_start": 0.05,
        "line_art_ratio_stop": 0.15,
        "chunk_size": 2048,
    }


def _segment_reliability(
    alpha_lr: np.ndarray,
    graph: object,
    *,
    hard_edge_gate: bool,
    line_art_bypass_enabled: bool,
    line_art_radius_limit: float,
    line_art_min_component: int,
    line_art_neighborhood: int,
    line_art_peak_min: float,
    line_art_max_foreground_area: int | None,
    line_art_max_canvas_area: int | None,
    line_art_ratio_start: float,
    line_art_ratio_stop: float,
) -> np.ndarray:
    if not hard_edge_gate:
        return np.ones(len(graph.starts), dtype=np.float32)

    gradient_x = sobel(alpha_lr, axis=1, mode="nearest") / 8.0
    gradient_y = sobel(alpha_lr, axis=0, mode="nearest") / 8.0
    gradient = np.hypot(gradient_x, gradient_y).astype(np.float32)
    local_kernel = np.full((5, 5), 1.0 / 25.0, dtype=np.float32)
    local_background = convolve((alpha_lr <= 0.05).astype(np.float32), local_kernel, mode="nearest")
    local_foreground = convolve((alpha_lr >= 0.95).astype(np.float32), local_kernel, mode="nearest")
    segment_midpoints = (graph.starts + graph.ends) * 0.5
    segment_gradient = _bilinear_sample(gradient, segment_midpoints)
    segment_binary_support = np.minimum(
        _bilinear_sample(local_background, segment_midpoints),
        _bilinear_sample(local_foreground, segment_midpoints),
    )
    gradient_score = np.clip((segment_gradient - 0.08) / 0.25, 0.0, 1.0)
    support_score = np.clip(segment_binary_support / 0.12, 0.0, 1.0)
    reliability = (gradient_score * support_score).astype(np.float32)

    if line_art_bypass_enabled:
        line_art_gate_lr = persistent_centerline_support_gate(
            alpha_lr,
            np.ones_like(alpha_lr),
            medial_radius_limit=line_art_radius_limit,
            min_component=line_art_min_component,
            neighborhood=line_art_neighborhood,
            peak_min=line_art_peak_min,
            max_foreground_area=line_art_max_foreground_area,
            max_canvas_area=line_art_max_canvas_area,
            global_ratio_start=line_art_ratio_start,
            global_ratio_stop=line_art_ratio_stop,
        )
        reliability = np.maximum(
            reliability,
            _bilinear_sample(line_art_gate_lr, segment_midpoints),
        ).astype(np.float32)
    return reliability


def segment_guided_normalized_lanczos_rgb(
    rgb_lr: np.ndarray,
    alpha_lr: np.ndarray,
    alpha_hr: np.ndarray,
    *,
    scale: int = 4,
    level: float = 0.5,
    nearest_segments: int = 12,
    support_radius_normal: float = 3.0,
    support_radius_tangent: float = 3.0,
    lanczos_lobes: int = 3,
    steer_kernel: bool = False,
    edge_band_hr: float = 3.5,
    condition_limit: float = 8.0,
    condition_soft_start: float = 2.0,
    arc_radius: float = 6.0,
    contour_gate_enabled: bool = True,
    alpha_sample_floor: float = 1.0 / 255.0,
    alpha_mass_floor: float = 0.03,
    local_range_epsilon: float = 0.01,
    max_residual: float = 0.20,
    strength: float = 1.0,
    hard_edge_gate: bool = True,
    line_art_bypass_enabled: bool = True,
    line_art_radius_limit: float = 0.75,
    line_art_min_component: int = 3,
    line_art_neighborhood: int = 3,
    line_art_peak_min: float = 0.8,
    line_art_max_foreground_area: int | None = 4096,
    line_art_max_canvas_area: int | None = 16384,
    line_art_ratio_start: float = 0.05,
    line_art_ratio_stop: float = 0.15,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, SegmentGuidedRGBDiagnostics]:
    """Refine alpha-edge RGB with a contour-aware normalized Lanczos residual.

    The candidate is reconstructed in linear light as ``sum(w*a*C)/sum(w*a)``.
    Hidden RGB from fully transparent samples therefore cannot leak into the
    result.  Samples may additionally be restricted to the same contour and a
    local arc-distance neighborhood.  The candidate is range-clamped, checked
    for signed-kernel conditioning, and blended into the premultiplied Lanczos
    base only inside the final SAL alpha boundary band.
    """

    rgb = _as_rgb_u8(rgb_lr)
    alpha_low = _as_alpha(alpha_lr, "alpha_lr")
    alpha_high = _as_alpha(alpha_hr, "alpha_hr")
    if alpha_low.shape != rgb.shape[:2]:
        raise ValueError("alpha_lr must match rgb_lr")
    expected_shape = (rgb.shape[0] * scale, rgb.shape[1] * scale)
    if alpha_high.shape != expected_shape:
        raise ValueError("alpha_hr must match rgb_lr scaled by scale")
    if not isinstance(scale, int) or scale < 1:
        raise ValueError("scale must be a positive integer")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between zero and one")
    if nearest_segments < 1 or lanczos_lobes < 1 or chunk_size < 1:
        raise ValueError("segment count, Lanczos lobes, and chunk size must be positive")
    if support_radius_normal <= 0.0 or support_radius_tangent <= 0.0:
        raise ValueError("support radii must be positive")
    if edge_band_hr <= 0.0 or arc_radius <= 0.0:
        raise ValueError("edge band and arc radius must be positive")
    if not 1.0 <= condition_soft_start < condition_limit:
        raise ValueError("condition thresholds require 1 <= soft_start < limit")
    if not 0.0 <= alpha_sample_floor < 1.0 or alpha_mass_floor <= 0.0:
        raise ValueError("alpha support thresholds are invalid")
    if local_range_epsilon < 0.0 or max_residual < 0.0:
        raise ValueError("range epsilon and max residual must be non-negative")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")

    base_u8 = resize_premultiplied_rgb_lanczos(rgb, alpha_low, scale=scale)
    base_linear = _srgb_to_linear(base_u8.astype(np.float32) / 255.0)
    graph = _extract_segment_graph(alpha_low, level=level)
    empty = np.zeros(expected_shape, dtype=np.float32)
    empty_bool = np.zeros(expected_shape, dtype=bool)
    if graph is None:
        diagnostics = SegmentGuidedRGBDiagnostics(
            gate_hr=empty,
            condition_hr=np.full(expected_shape, np.inf, dtype=np.float32),
            contour_support_hr=empty,
            residual_luma_hr=empty,
            active_mask_hr=empty_bool,
        )
        return base_u8, diagnostics

    hr_query = _hr_queries(expected_shape, scale)
    hr_segment_indices, _, hr_distance_lr = _query_segment_graph(
        graph,
        hr_query,
        nearest_segments=nearest_segments,
    )
    segment_reliability = _segment_reliability(
        alpha_low,
        graph,
        hard_edge_gate=hard_edge_gate,
        line_art_bypass_enabled=line_art_bypass_enabled,
        line_art_radius_limit=line_art_radius_limit,
        line_art_min_component=line_art_min_component,
        line_art_neighborhood=line_art_neighborhood,
        line_art_peak_min=line_art_peak_min,
        line_art_max_foreground_area=line_art_max_foreground_area,
        line_art_max_canvas_area=line_art_max_canvas_area,
        line_art_ratio_start=line_art_ratio_start,
        line_art_ratio_stop=line_art_ratio_stop,
    )
    frame_reliability = segment_reliability[hr_segment_indices]

    alpha_flat_hr = alpha_high.ravel()
    # Peak at alpha=0.5, smoothly decreasing toward fully transparent/opaque.
    partial_alpha_gate = np.sqrt(
        np.clip(4.0 * alpha_flat_hr * (1.0 - alpha_flat_hr), 0.0, 1.0)
    ).astype(np.float32)
    active = np.flatnonzero(
        (hr_distance_lr * float(scale) <= edge_band_hr)
        & (alpha_flat_hr > 1e-6)
        & (alpha_flat_hr < 1.0 - 1e-6)
        & (partial_alpha_gate > 1e-4)
        & (frame_reliability > 1e-4)
    )

    output_linear = base_linear.reshape(-1, 3).copy()
    gate_flat = np.zeros(alpha_high.size, dtype=np.float32)
    condition_flat = np.full(alpha_high.size, np.inf, dtype=np.float32)
    contour_support_flat = np.zeros(alpha_high.size, dtype=np.float32)
    residual_luma_flat = np.zeros(alpha_high.size, dtype=np.float32)
    if len(active) == 0:
        output_linear[alpha_flat_hr <= 1e-6] = 0.0
        output_u8 = (
            np.rint(np.clip(_linear_to_srgb(np.clip(output_linear, 0.0, 1.0)), 0.0, 1.0) * 255.0)
            .astype(np.uint8)
            .reshape((*expected_shape, 3))
        )
        diagnostics = SegmentGuidedRGBDiagnostics(
            gate_hr=gate_flat.reshape(expected_shape),
            condition_hr=condition_flat.reshape(expected_shape),
            contour_support_hr=contour_support_flat.reshape(expected_shape),
            residual_luma_hr=residual_luma_flat.reshape(expected_shape),
            active_mask_hr=(gate_flat.reshape(expected_shape) > 0.0),
        )
        return output_u8, diagnostics

    lr_height, lr_width = alpha_low.shape
    lr_y, lr_x = np.mgrid[0:lr_height, 0:lr_width]
    lr_query = np.stack([lr_y.ravel(), lr_x.ravel()], axis=1).astype(np.float32)
    lr_segment_indices, _, _ = _query_segment_graph(
        graph,
        lr_query,
        nearest_segments=nearest_segments,
    )

    rgb_linear = _srgb_to_linear(rgb.astype(np.float32) / 255.0)
    rgb_flat = rgb_linear.reshape(-1, 3)
    alpha_flat_lr = alpha_low.ravel()
    support = int(np.ceil(max(support_radius_normal, support_radius_tangent)))
    offset_y, offset_x = np.mgrid[-support : support + 1, -support : support + 1]
    offset_y = offset_y.ravel().astype(np.int64)
    offset_x = offset_x.ravel().astype(np.int64)

    for start in range(0, len(active), chunk_size):
        target_indices = active[start : start + chunk_size]
        points = hr_query[target_indices]
        target_segments = hr_segment_indices[target_indices]

        if steer_kernel:
            tangents, coherence = _query_segment_frame(
                graph,
                points,
                nearest_segments=nearest_segments,
            )
            tangent_support = (
                float(support_radius_normal)
                + coherence * float(support_radius_tangent - support_radius_normal)
            )[:, None]
        else:
            tangents = np.zeros((len(points), 2), dtype=np.float32)
            tangents[:, 1] = 1.0
            tangent_support = np.full(
                (len(points), 1), float(support_radius_tangent), dtype=np.float32
            )
        normals = np.stack((-tangents[:, 1], tangents[:, 0]), axis=1)

        center_y = np.floor(points[:, 0]).astype(np.int64)
        center_x = np.floor(points[:, 1]).astype(np.int64)
        sample_y = center_y[:, None] + offset_y[None, :]
        sample_x = center_x[:, None] + offset_x[None, :]
        valid = (sample_y >= 0) & (sample_y < lr_height) & (sample_x >= 0) & (sample_x < lr_width)
        safe_y = np.clip(sample_y, 0, lr_height - 1)
        safe_x = np.clip(sample_x, 0, lr_width - 1)
        sample_flat = safe_y * lr_width + safe_x

        delta_y = sample_y.astype(np.float32) - points[:, 0:1]
        delta_x = sample_x.astype(np.float32) - points[:, 1:2]
        normal_coordinate = delta_y * normals[:, 0:1] + delta_x * normals[:, 1:2]
        tangent_coordinate = delta_y * tangents[:, 0:1] + delta_x * tangents[:, 1:2]
        raw_weights = _lanczos_weight(
            float(lanczos_lobes) * normal_coordinate / float(support_radius_normal),
            lanczos_lobes,
        ) * _lanczos_weight(
            float(lanczos_lobes) * tangent_coordinate / tangent_support,
            lanczos_lobes,
        )
        raw_weights *= valid

        contour_gate = valid.astype(np.float32)
        if contour_gate_enabled:
            sample_segments = lr_segment_indices[sample_flat]
            target_contours = graph.contour_ids[target_segments][:, None]
            sample_contours = graph.contour_ids[sample_segments]
            same_contour = sample_contours == target_contours
            arc_delta = np.abs(
                graph.arc_midpoints[sample_segments] - graph.arc_midpoints[target_segments][:, None]
            )
            contour_lengths = graph.contour_lengths[target_contours]
            closed = graph.contour_closed[target_contours]
            arc_delta = np.where(
                closed,
                np.minimum(arc_delta, np.maximum(contour_lengths - arc_delta, 0.0)),
                arc_delta,
            )
            contour_gate = np.where(
                same_contour,
                np.exp(-0.5 * (arc_delta / float(arc_radius)) ** 2),
                0.0,
            ).astype(np.float32)

        sample_alpha = alpha_flat_lr[sample_flat]
        visible = sample_alpha >= float(alpha_sample_floor)
        gated_weights = raw_weights * contour_gate * visible
        alpha_weights = gated_weights * sample_alpha
        signed_mass = np.sum(alpha_weights, axis=1)
        absolute_mass = np.sum(np.abs(alpha_weights), axis=1)
        condition = absolute_mass / np.maximum(np.abs(signed_mass), 1e-8)
        condition_flat[target_indices] = condition.astype(np.float32)

        raw_alpha_mass = np.sum(np.abs(raw_weights) * sample_alpha * visible, axis=1)
        contour_support = absolute_mass / np.maximum(raw_alpha_mass, 1e-8)
        contour_support = np.clip(contour_support, 0.0, 1.0).astype(np.float32)
        contour_support_flat[target_indices] = contour_support

        stable = (
            (np.abs(signed_mass) >= float(alpha_mass_floor))
            & (absolute_mass >= float(alpha_mass_floor))
            & (condition <= float(condition_limit))
            & (contour_support > 1e-4)
        )
        if not stable.any():
            continue

        numerator = np.sum(
            alpha_weights[..., None] * rgb_flat[sample_flat],
            axis=1,
        )
        candidate = np.zeros((len(target_indices), 3), dtype=np.float32)
        candidate[stable] = numerator[stable] / signed_mass[stable, None]

        members = valid & visible & (contour_gate > 1e-5) & (np.abs(raw_weights) > 1e-8)
        sampled_rgb = rgb_flat[sample_flat]
        local_minimum = np.min(
            np.where(members[..., None], sampled_rgb, np.inf),
            axis=1,
        )
        local_maximum = np.max(
            np.where(members[..., None], sampled_rgb, -np.inf),
            axis=1,
        )
        finite_range = np.isfinite(local_minimum).all(axis=1) & np.isfinite(local_maximum).all(
            axis=1
        )
        stable &= finite_range
        candidate[stable] = np.clip(
            candidate[stable],
            local_minimum[stable] - float(local_range_epsilon),
            local_maximum[stable] + float(local_range_epsilon),
        )

        base_chunk = output_linear[target_indices]
        residual = np.clip(
            candidate - base_chunk,
            -float(max_residual),
            float(max_residual),
        )
        condition_gate = 1.0 - _smoothstep(
            (condition - float(condition_soft_start))
            / float(condition_limit - condition_soft_start)
        )
        gate = (
            condition_gate
            * contour_support
            * partial_alpha_gate[target_indices]
            * frame_reliability[target_indices]
            * float(strength)
        ).astype(np.float32)
        gate[~stable] = 0.0
        output_linear[target_indices] = np.clip(
            base_chunk + residual * gate[:, None],
            0.0,
            1.0,
        )
        gate_flat[target_indices] = gate
        residual_luma_flat[target_indices] = np.sum(
            np.abs(residual) * np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32),
            axis=1,
        )

    output_linear[alpha_flat_hr <= 1e-6] = 0.0
    output_srgb = np.clip(_linear_to_srgb(np.clip(output_linear, 0.0, 1.0)), 0.0, 1.0)
    output_u8 = np.rint(output_srgb * 255.0).astype(np.uint8).reshape((*expected_shape, 3))
    diagnostics = SegmentGuidedRGBDiagnostics(
        gate_hr=gate_flat.reshape(expected_shape),
        condition_hr=condition_flat.reshape(expected_shape),
        contour_support_hr=contour_support_flat.reshape(expected_shape),
        residual_luma_hr=residual_luma_flat.reshape(expected_shape),
        active_mask_hr=(gate_flat.reshape(expected_shape) > 0.0),
    )
    return output_u8, diagnostics


def upscale_rgba_sal_classic_x4(
    rgba_lr: np.ndarray,
    *,
    rgb_options: dict[str, object] | None = None,
) -> tuple[np.ndarray, SegmentGuidedRGBDiagnostics]:
    """Upscale straight RGBA x4 with the complete SAL-Classic route."""

    rgba = np.asarray(rgba_lr)
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
        raise ValueError("rgba_lr must be a uint8 HxWx4 array")
    alpha_lr = rgba[..., 3].astype(np.float32) / 255.0
    alpha_hr = resize_alpha_main_x4(alpha_lr)
    options = main_sal_classic_rgb_options()
    if rgb_options is not None:
        unknown = set(rgb_options) - set(options)
        if unknown:
            raise ValueError(f"unknown SAL-Classic RGB options: {sorted(unknown)}")
        options.update(rgb_options)
    rgb_hr, diagnostics = segment_guided_normalized_lanczos_rgb(
        rgba[..., :3],
        alpha_lr,
        alpha_hr,
        scale=4,
        **options,
    )
    alpha_u8 = np.rint(np.clip(alpha_hr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.dstack((rgb_hr, alpha_u8)), diagnostics
