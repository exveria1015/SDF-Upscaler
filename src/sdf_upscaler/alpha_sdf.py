"""Deterministic signed-distance-field baseline for hard alpha assets.

This is deliberately not presented as a learned alpha model.  It establishes the
minimum result that a later residual Alpha-SDF head must beat.  The method is
appropriate for hard boundaries such as UI, logos, glyphs, and sprites.  It is
not an adequate representation for hair, smoke, glow, or glass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import (
    convolve,
    distance_transform_edt,
    label,
    maximum_filter,
    sobel,
    uniform_filter,
)
from scipy.spatial import cKDTree
from skimage.measure import find_contours
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class _SegmentGraph:
    starts: np.ndarray
    ends: np.ndarray
    tangents: np.ndarray
    contour_ids: np.ndarray
    arc_midpoints: np.ndarray
    contour_lengths: np.ndarray
    contour_closed: np.ndarray
    tree: cKDTree


@dataclass(frozen=True)
class AlphaCalibrationGateDiagnostics:
    """Low-resolution evidence and final gate for calibrated alpha protection."""

    broad_soft_score_lr: np.ndarray
    broad_soft_gate_lr: np.ndarray
    reprojection_error_base_lr: np.ndarray
    reprojection_error_candidate_lr: np.ndarray
    reprojection_gain_lr: np.ndarray
    reprojection_gate_lr: np.ndarray
    support_score_lr: np.ndarray
    support_gate_lr: np.ndarray
    centerline_support_gate_lr: np.ndarray
    combined_gate_hr: np.ndarray


def _as_alpha_array(alpha: np.ndarray) -> np.ndarray:
    array = np.asarray(alpha, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"alpha must be a 2D array, got shape={array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("alpha contains NaN or infinity")
    if array.size == 0:
        raise ValueError("alpha must not be empty")
    return np.clip(array, 0.0, 1.0)


def alpha_to_sdf(alpha: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert alpha coverage to an SDF measured in low-resolution pixels.

    Positive values are inside the foreground.  A half-pixel correction places
    the zero crossing between adjacent foreground/background pixel centers.
    Constant mattes use signed infinity because no boundary exists; the inverse
    conversion handles these values as exact zero/one coverage.
    """

    alpha_array = _as_alpha_array(alpha)
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")

    inside = alpha_array >= threshold
    if inside.all():
        return np.full_like(alpha_array, np.inf)
    if not inside.any():
        return np.full_like(alpha_array, -np.inf)

    distance_inside = distance_transform_edt(inside)
    distance_outside = distance_transform_edt(~inside)
    sdf = np.where(
        inside,
        distance_inside - 0.5,
        -(distance_outside - 0.5),
    )
    return sdf.astype(np.float32, copy=False)


def sdf_to_alpha(sdf: np.ndarray, feather_width: float = 1.0) -> np.ndarray:
    """Rasterize an SDF to linear coverage around its zero crossing."""

    sdf_array = np.asarray(sdf, dtype=np.float32)
    if sdf_array.ndim != 2 or sdf_array.size == 0:
        raise ValueError(f"sdf must be a non-empty 2D array, got shape={sdf_array.shape}")
    if feather_width <= 0:
        raise ValueError("feather_width must be positive")

    coverage = 0.5 + sdf_array / feather_width
    return np.clip(coverage, 0.0, 1.0).astype(np.float32, copy=False)


def _resize_float_plane(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(array, dtype=np.float32), mode="F")
    resized = image.resize(size, resample=Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.float32)


def resize_alpha_lanczos(alpha: np.ndarray, scale: int = 4) -> np.ndarray:
    """Upscale alpha as floating-point Lanczos coverage."""

    alpha_array = _as_alpha_array(alpha)
    if not isinstance(scale, int) or scale < 1:
        raise ValueError("scale must be a positive integer")
    height, width = alpha_array.shape
    image = Image.fromarray(alpha_array, mode="F")
    resized = image.resize(
        (width * scale, height * scale),
        resample=Image.Resampling.LANCZOS,
    )
    return np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0)


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    return (clipped * clipped * (3.0 - 2.0 * clipped)).astype(np.float32)


def spatial_support_gate(
    confidence_lr: np.ndarray,
    *,
    window: int = 5,
    start: float = 0.08,
    stop: float = 0.30,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a continuous gate for spatially supported LR confidence.

    Isolated positive evidence receives a lower score than a coherent band or
    region.  The operation is deliberately continuous so that the support term
    can be calibrated without connected-component hard thresholds.
    """

    confidence = _as_alpha_array(confidence_lr)
    if window < 1:
        raise ValueError("support window must be positive")
    if not 0.0 <= start < stop <= 1.0:
        raise ValueError("support thresholds must satisfy 0 <= start < stop <= 1")
    score = uniform_filter(confidence, size=int(window), mode="nearest").astype(np.float32)
    gate = _smoothstep((score - float(start)) / float(stop - start))
    return gate, score


def persistent_centerline_support_gate(
    alpha_lr: np.ndarray,
    confidence_lr: np.ndarray,
    *,
    medial_radius_limit: float = 0.75,
    min_component: int = 5,
    neighborhood: int = 3,
    peak_min: float = 0.8,
    peak_stop: float | None = None,
    max_foreground_area: int | None = None,
    max_canvas_area: int | None = None,
    global_ratio_start: float | None = None,
    global_ratio_stop: float | None = None,
) -> np.ndarray:
    """Keep confidence near persistent thin centerlines.

    A square area average suppresses a valid one-pixel stroke because the
    evidence occupies only a small fraction of the window.  This bypass is
    deliberately restricted to 8-connected thin skeleton components with a
    minimum LR persistence, so isolated shape tips do not regain full weight.
    It never creates confidence: the retained centerline mask is multiplied by
    the original local confidence.
    """

    alpha_array = _as_alpha_array(alpha_lr)
    confidence = _as_alpha_array(confidence_lr)
    if alpha_array.shape != confidence.shape:
        raise ValueError("alpha and confidence must have identical shapes")
    if medial_radius_limit <= 0:
        raise ValueError("medial radius limit must be positive")
    if min_component < 1 or neighborhood < 1:
        raise ValueError("component and neighborhood sizes must be positive")
    if not 0.5 < peak_min <= 1.0:
        raise ValueError("centerline peak minimum must be in (0.5, 1]")
    if peak_stop is not None and not peak_min < peak_stop <= 1.0:
        raise ValueError("centerline peak stop must be above peak minimum and at most one")
    if max_foreground_area is not None and max_foreground_area < 1:
        raise ValueError("maximum foreground area must be positive")
    if max_canvas_area is not None and max_canvas_area < 1:
        raise ValueError("maximum canvas area must be positive")
    if (global_ratio_start is None) != (global_ratio_stop is None):
        raise ValueError("global ratio thresholds must both be set or both be omitted")
    if global_ratio_start is not None and not (
        0.0 <= global_ratio_start < float(global_ratio_stop) <= 1.0
    ):
        raise ValueError("global ratio thresholds must satisfy 0 <= start < stop <= 1")
    mask = alpha_array >= 0.5
    peak = float(np.max(alpha_array))
    if (
        not mask.any()
        or mask.all()
        or peak < float(peak_min)
        or (peak_stop is not None and peak >= float(peak_stop))
        or (max_foreground_area is not None and int(np.sum(mask)) > int(max_foreground_area))
        or (max_canvas_area is not None and alpha_array.size > int(max_canvas_area))
    ):
        return np.zeros_like(confidence)
    skeleton = skeletonize(mask)
    global_gate = 0.0
    if global_ratio_start is not None:
        skeleton_components, skeleton_component_count = label(
            skeleton,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        skeleton_sizes = (
            np.bincount(skeleton_components.ravel())[1:]
            if skeleton_component_count
            else np.empty(0, dtype=np.int64)
        )
        if skeleton_sizes.size and int(np.max(skeleton_sizes)) >= int(min_component):
            # A thick but character-like glyph can have no radius<=limit
            # samples, while still being topologically line-like.  Natural
            # filled silhouettes have a much smaller skeleton/area ratio.
            skeleton_ratio = float(np.sum(skeleton)) / max(float(np.sum(mask)), 1.0)
            global_gate = float(
                _smoothstep(
                    np.asarray(
                        (skeleton_ratio - global_ratio_start)
                        / (float(global_ratio_stop) - global_ratio_start),
                        dtype=np.float32,
                    )
                )
            )
    radius = np.maximum(distance_transform_edt(mask).astype(np.float32) - 0.5, 0.5)
    thin = skeleton & (radius <= float(medial_radius_limit))
    components, component_count = label(thin, structure=np.ones((3, 3), dtype=np.uint8))
    if component_count == 0:
        return np.full_like(confidence, global_gate)
    sizes = np.bincount(components.ravel())
    keep = sizes >= int(min_component)
    keep[0] = False
    persistent = keep[components]
    neighborhood_mask = maximum_filter(
        persistent.astype(np.float32),
        size=int(neighborhood),
        mode="nearest",
    )
    local_gate = neighborhood_mask * confidence
    if global_ratio_start is not None:
        local_gate = np.maximum(local_gate, global_gate)
    return np.clip(local_gate, 0.0, 1.0).astype(np.float32)


def broad_soft_reprojection_gate(
    alpha_lr: np.ndarray,
    base_hr: np.ndarray,
    candidate_hr: np.ndarray,
    *,
    scale: int = 4,
    broad_soft_window: int = 5,
    broad_soft_start: float = 0.16,
    broad_soft_stop: float = 0.52,
    reprojection_window: int = 5,
    reprojection_margin: float = 0.0,
    reprojection_transition: float = 0.001,
    support_gate_enabled: bool = False,
    support_window: int = 9,
    support_start: float = 0.15,
    support_stop: float = 0.50,
    support_centerline_bypass_enabled: bool = False,
    support_centerline_radius_limit: float = 0.75,
    support_centerline_min_component: int = 3,
    support_centerline_neighborhood: int = 3,
    support_centerline_peak_min: float = 0.8,
    support_centerline_peak_stop: float | None = None,
    support_centerline_max_foreground_area: int | None = 4096,
    support_centerline_ratio_start: float = 0.10,
    support_centerline_ratio_stop: float = 0.30,
) -> tuple[np.ndarray, AlphaCalibrationGateDiagnostics]:
    """Gate geometry corrections using softness and LR self-consistency.

    The broad-soft term suppresses Segment-SDF on large intermediate-alpha
    regions while retaining narrow antialiased boundaries.  The reprojection
    term permits a correction only where Lanczos-downsampling the candidate
    reproduces the observed LR alpha more accurately than the Lanczos base.
    An optional spatial-support term rejects isolated positive reprojection
    evidence.  Terms are computed locally in LR space, smoothly upsampled, and
    multiplied in HR space.  This is a calibration gate, not a learned model.
    """

    alpha_array = _as_alpha_array(alpha_lr)
    base_array = _as_alpha_array(base_hr)
    candidate_array = _as_alpha_array(candidate_hr)
    if not isinstance(scale, int) or scale < 1:
        raise ValueError("scale must be a positive integer")
    expected_shape = (alpha_array.shape[0] * scale, alpha_array.shape[1] * scale)
    if base_array.shape != expected_shape or candidate_array.shape != expected_shape:
        raise ValueError(
            "base_hr and candidate_hr must match alpha_lr scaled by scale: "
            f"expected {expected_shape}, got {base_array.shape} and {candidate_array.shape}"
        )
    if broad_soft_window < 1 or reprojection_window < 1:
        raise ValueError("gate windows must be positive")
    if not 0.0 <= broad_soft_start < broad_soft_stop <= 1.0:
        raise ValueError("broad-soft thresholds must satisfy 0 <= start < stop <= 1")
    if reprojection_margin < 0.0 or reprojection_transition <= 0.0:
        raise ValueError("reprojection margin must be non-negative and transition positive")

    intermediate_mass = 4.0 * alpha_array * (1.0 - alpha_array)
    broad_soft_score = uniform_filter(
        intermediate_mass.astype(np.float32),
        size=int(broad_soft_window),
        mode="nearest",
    )
    broad_position = (broad_soft_score - float(broad_soft_start)) / float(
        broad_soft_stop - broad_soft_start
    )
    broad_soft_gate = 1.0 - _smoothstep(broad_position)

    lr_size = (alpha_array.shape[1], alpha_array.shape[0])
    base_reprojected = np.asarray(
        Image.fromarray(base_array, mode="F").resize(lr_size, Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    candidate_reprojected = np.asarray(
        Image.fromarray(candidate_array, mode="F").resize(lr_size, Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    base_error = uniform_filter(
        np.abs(base_reprojected - alpha_array),
        size=int(reprojection_window),
        mode="nearest",
    ).astype(np.float32)
    candidate_error = uniform_filter(
        np.abs(candidate_reprojected - alpha_array),
        size=int(reprojection_window),
        mode="nearest",
    ).astype(np.float32)
    gain = (base_error - candidate_error).astype(np.float32)
    gain_position = (gain - float(reprojection_margin)) / float(reprojection_transition)
    reprojection_gate = _smoothstep(gain_position)

    support_gate, support_score = spatial_support_gate(
        reprojection_gate,
        window=support_window,
        start=support_start,
        stop=support_stop,
    )
    if not support_gate_enabled:
        support_gate = np.ones_like(reprojection_gate)
    centerline_support_gate = persistent_centerline_support_gate(
        alpha_array,
        reprojection_gate,
        medial_radius_limit=support_centerline_radius_limit,
        min_component=support_centerline_min_component,
        neighborhood=support_centerline_neighborhood,
        peak_min=support_centerline_peak_min,
        peak_stop=support_centerline_peak_stop,
        max_foreground_area=support_centerline_max_foreground_area,
        global_ratio_start=(
            support_centerline_ratio_start if support_centerline_bypass_enabled else None
        ),
        global_ratio_stop=(
            support_centerline_ratio_stop if support_centerline_bypass_enabled else None
        ),
    )
    if not support_centerline_bypass_enabled:
        centerline_support_gate = np.zeros_like(reprojection_gate)

    output_size = (expected_shape[1], expected_shape[0])
    broad_hr = np.asarray(
        Image.fromarray(broad_soft_gate.astype(np.float32), mode="F").resize(
            output_size, Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    reprojection_hr = np.asarray(
        Image.fromarray(reprojection_gate.astype(np.float32), mode="F").resize(
            output_size, Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    support_hr = np.asarray(
        Image.fromarray(support_gate.astype(np.float32), mode="F").resize(
            output_size, Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    centerline_support_hr = np.asarray(
        Image.fromarray(centerline_support_gate.astype(np.float32), mode="F").resize(
            output_size, Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    combined_hr = np.clip(
        np.maximum(broad_hr * reprojection_hr * support_hr, centerline_support_hr),
        0.0,
        1.0,
    ).astype(np.float32)
    diagnostics = AlphaCalibrationGateDiagnostics(
        broad_soft_score_lr=broad_soft_score.astype(np.float32),
        broad_soft_gate_lr=broad_soft_gate.astype(np.float32),
        reprojection_error_base_lr=base_error,
        reprojection_error_candidate_lr=candidate_error,
        reprojection_gain_lr=gain,
        reprojection_gate_lr=reprojection_gate,
        support_score_lr=support_score,
        support_gate_lr=support_gate,
        centerline_support_gate_lr=centerline_support_gate,
        combined_gate_hr=combined_hr,
    )
    return combined_hr, diagnostics


def circle_segment_fraction(normalized_distance: np.ndarray) -> np.ndarray:
    """Area fraction of a circle on the positive side of an offset line."""

    x = np.clip(np.asarray(normalized_distance, dtype=np.float64), -1.0, 1.0)
    root = np.sqrt(np.maximum(0.0, 1.0 - x * x))
    fraction = (np.arccos(-x) + x * root) / np.pi
    return fraction.astype(np.float32)


def invert_circle_segment_fraction(
    coverage: np.ndarray,
    *,
    iterations: int = 6,
) -> np.ndarray:
    """Invert circular-segment coverage with bounded Newton iterations."""

    fraction = np.clip(np.asarray(coverage, dtype=np.float64), 0.0, 1.0)
    if iterations < 1:
        raise ValueError("iterations must be positive")

    x = np.sin(np.pi * (fraction - 0.5))
    active = np.logical_and(fraction > 0.0, fraction < 1.0)
    for _ in range(iterations):
        clipped = np.clip(x, -0.999999, 0.999999)
        root = np.sqrt(np.maximum(1e-12, 1.0 - clipped * clipped))
        estimate = (np.arccos(-clipped) + clipped * root) / np.pi
        derivative = 2.0 * root / np.pi
        step = np.clip((estimate - fraction) / derivative, -0.5, 0.5)
        x = np.where(active, np.clip(clipped - step, -0.999999, 0.999999), x)

    x = np.where(fraction <= 0.0, -1.0, x)
    x = np.where(fraction >= 1.0, 1.0, x)
    return x.astype(np.float32)


def _disk_kernel(radius: float) -> np.ndarray:
    if radius < 1.0:
        raise ValueError("circle radius must be at least one pixel")
    extent = int(np.ceil(radius))
    yy, xx = np.mgrid[-extent : extent + 1, -extent : extent + 1]
    kernel = (xx * xx + yy * yy <= radius * radius).astype(np.float32)
    kernel /= float(kernel.sum())
    return kernel


def circle_newton_sdf(
    alpha: np.ndarray,
    *,
    radius: float = 2.5,
    iterations: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate contour distance and confidence from local circular support.

    The alpha mass in a disk is interpreted as a circular-segment area. Newton
    inversion gives the signed center-to-contour distance. Structure-tensor
    coherence is returned as confidence in the local straight-contour model.
    """

    alpha_array = _as_alpha_array(alpha)
    kernel = _disk_kernel(radius)
    local_fraction = convolve(alpha_array, kernel, mode="nearest")
    normalized_distance = invert_circle_segment_fraction(
        local_fraction,
        iterations=iterations,
    )
    distance = normalized_distance * float(radius)

    grad_x = sobel(alpha_array, axis=1, mode="nearest") / 8.0
    grad_y = sobel(alpha_array, axis=0, mode="nearest") / 8.0
    j_xx = convolve(grad_x * grad_x, kernel, mode="nearest")
    j_yy = convolve(grad_y * grad_y, kernel, mode="nearest")
    j_xy = convolve(grad_x * grad_y, kernel, mode="nearest")
    numerator = np.sqrt(np.maximum(0.0, (j_xx - j_yy) ** 2 + 4.0 * j_xy**2))
    coherence = numerator / (j_xx + j_yy + 1e-8)

    # Circular averaging is allowed to refine offset, not invent topology.
    sign = np.where(alpha_array >= 0.5, 1.0, -1.0)
    distance = np.abs(distance) * sign
    return distance.astype(np.float32), np.clip(coherence, 0.0, 1.0).astype(np.float32)


def _extract_segment_graph(alpha: np.ndarray, *, level: float) -> _SegmentGraph | None:
    contours = find_contours(
        alpha,
        level=level,
        fully_connected="high",
        positive_orientation="high",
    )
    starts_parts: list[np.ndarray] = []
    ends_parts: list[np.ndarray] = []
    tangent_parts: list[np.ndarray] = []
    contour_id_parts: list[np.ndarray] = []
    arc_parts: list[np.ndarray] = []
    contour_lengths: list[float] = []
    contour_closed: list[bool] = []
    for contour in contours:
        if len(contour) < 2:
            continue
        starts = contour[:-1].astype(np.float32)
        ends = contour[1:].astype(np.float32)
        vectors = ends - starts
        lengths = np.linalg.norm(vectors, axis=1)
        valid = lengths > 1e-8
        if not valid.any():
            continue
        starts = starts[valid]
        ends = ends[valid]
        vectors = vectors[valid]
        lengths = lengths[valid]
        tangents = vectors / lengths[:, None]
        cumulative = np.concatenate(([0.0], np.cumsum(lengths[:-1], dtype=np.float64)))
        arc_midpoints = cumulative + 0.5 * lengths
        total_length = float(np.sum(lengths))
        is_closed = bool(np.linalg.norm(contour[0] - contour[-1]) <= 1e-5)

        starts_parts.append(starts)
        ends_parts.append(ends)
        tangent_parts.append(tangents.astype(np.float32))
        contour_id_parts.append(np.full(len(starts), len(contour_lengths), dtype=np.int32))
        arc_parts.append(arc_midpoints.astype(np.float32))
        contour_lengths.append(total_length)
        contour_closed.append(is_closed)

    if not starts_parts:
        return None
    starts = np.concatenate(starts_parts, axis=0).astype(np.float32, copy=False)
    ends = np.concatenate(ends_parts, axis=0).astype(np.float32, copy=False)
    tangents = np.concatenate(tangent_parts, axis=0).astype(np.float32, copy=False)
    contour_ids = np.concatenate(contour_id_parts, axis=0)
    arc_midpoints = np.concatenate(arc_parts, axis=0).astype(np.float32, copy=False)
    return _SegmentGraph(
        starts=starts,
        ends=ends,
        tangents=tangents,
        contour_ids=contour_ids,
        arc_midpoints=arc_midpoints,
        contour_lengths=np.asarray(contour_lengths, dtype=np.float32),
        contour_closed=np.asarray(contour_closed, dtype=bool),
        tree=cKDTree((starts + ends) * 0.5),
    )


def _query_segment_graph(
    graph: _SegmentGraph,
    points: np.ndarray,
    *,
    nearest_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact nearest segment index, closest point, and distance."""

    query = np.asarray(points, dtype=np.float32)
    if query.ndim != 2 or query.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {query.shape}")
    k = min(nearest_segments, len(graph.starts))
    _, indices = graph.tree.query(query, k=k, workers=-1)
    if k == 1:
        indices = indices[:, None]
    starts = graph.starts[indices]
    ends = graph.ends[indices]
    vectors = ends - starts
    offsets = query[:, None, :] - starts
    denominators = np.sum(vectors * vectors, axis=2)
    projections = np.sum(offsets * vectors, axis=2) / np.maximum(denominators, 1e-12)
    projections = np.clip(projections, 0.0, 1.0)
    closest = starts + projections[..., None] * vectors
    squared_distances = np.sum((query[:, None, :] - closest) ** 2, axis=2)
    candidate = np.argmin(squared_distances, axis=1)
    rows = np.arange(len(query))
    best_indices = indices[rows, candidate].astype(np.int64, copy=False)
    best_closest = closest[rows, candidate].astype(np.float32, copy=False)
    best_distance = np.sqrt(squared_distances[rows, candidate]).astype(np.float32, copy=False)
    return best_indices, best_closest, best_distance


def _hr_queries(shape: tuple[int, int], scale: int) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return np.stack(
        [
            (yy.ravel() + 0.5) / float(scale) - 0.5,
            (xx.ravel() + 0.5) / float(scale) - 0.5,
        ],
        axis=1,
    ).astype(np.float32)


def _lanczos_weight(value: np.ndarray, lobes: int) -> np.ndarray:
    coordinate = np.asarray(value, dtype=np.float32)
    weight = np.sinc(coordinate) * np.sinc(coordinate / float(lobes))
    return np.where(np.abs(coordinate) < float(lobes), weight, 0.0).astype(np.float32)


def _bilinear_sample(array: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Sample a 2D plane at ``(y, x)`` points with edge-clamped bilinear interpolation."""

    plane = np.asarray(array, dtype=np.float32)
    query = np.asarray(points, dtype=np.float32)
    y = np.clip(query[:, 0], 0.0, plane.shape[0] - 1.0)
    x = np.clip(query[:, 1], 0.0, plane.shape[1] - 1.0)
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.minimum(y0 + 1, plane.shape[0] - 1)
    x1 = np.minimum(x0 + 1, plane.shape[1] - 1)
    fy = y - y0
    fx = x - x0
    return (
        plane[y0, x0] * (1.0 - fy) * (1.0 - fx)
        + plane[y0, x1] * (1.0 - fy) * fx
        + plane[y1, x0] * fy * (1.0 - fx)
        + plane[y1, x1] * fy * fx
    ).astype(np.float32)


def _planar_segment_coverage(
    signed_distance_lr: np.ndarray,
    normals: np.ndarray,
    *,
    pixel_size_lr: float,
    feather_scale: float = 1.0,
) -> np.ndarray:
    """Approximate half-plane coverage using the projected square-pixel width."""

    if pixel_size_lr <= 0 or feather_scale <= 0:
        raise ValueError("pixel_size_lr and feather_scale must be positive")
    normal_array = np.asarray(normals, dtype=np.float32)
    projected_width = (
        float(pixel_size_lr)
        * (np.abs(normal_array[..., 0]) + np.abs(normal_array[..., 1]))
        * float(feather_scale)
    )
    coverage = 0.5 + np.asarray(signed_distance_lr, dtype=np.float32) / np.maximum(
        projected_width, 1e-6
    )
    return np.clip(coverage, 0.0, 1.0).astype(np.float32)


def _query_segment_frame(
    graph: _SegmentGraph,
    points: np.ndarray,
    *,
    nearest_segments: int,
    coherence_sigma: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate an unsigned tangent frame and angular coherence from nearby segments."""

    query = np.asarray(points, dtype=np.float32)
    k = min(nearest_segments, len(graph.starts))
    _, indices = graph.tree.query(query, k=k, workers=-1)
    if k == 1:
        indices = indices[:, None]
    starts = graph.starts[indices]
    ends = graph.ends[indices]
    vectors = ends - starts
    offsets = query[:, None, :] - starts
    denominators = np.sum(vectors * vectors, axis=2)
    projections = np.sum(offsets * vectors, axis=2) / np.maximum(denominators, 1e-12)
    projections = np.clip(projections, 0.0, 1.0)
    closest = starts + projections[..., None] * vectors
    distances = np.linalg.norm(query[:, None, :] - closest, axis=2)
    relative_distance = distances - np.min(distances, axis=1, keepdims=True)
    weights = np.exp(-0.5 * (relative_distance / float(coherence_sigma)) ** 2)

    tangents = graph.tangents[indices]
    tangent_y = tangents[..., 0]
    tangent_x = tangents[..., 1]
    double_cosine = tangent_x * tangent_x - tangent_y * tangent_y
    double_sine = 2.0 * tangent_x * tangent_y
    z_real = np.sum(weights * double_cosine, axis=1)
    z_imaginary = np.sum(weights * double_sine, axis=1)
    weight_sum = np.sum(weights, axis=1)
    coherence = np.sqrt(z_real * z_real + z_imaginary * z_imaginary) / np.maximum(weight_sum, 1e-8)
    angle = 0.5 * np.arctan2(z_imaginary, z_real)
    frame = np.stack((np.sin(angle), np.cos(angle)), axis=1).astype(np.float32)
    return frame, np.clip(coherence, 0.0, 1.0).astype(np.float32)


def resize_alpha_via_segments(
    alpha: np.ndarray,
    scale: int = 4,
    *,
    level: float = 0.5,
    feather_width_hr: float = 1.0,
    nearest_segments: int = 12,
) -> np.ndarray:
    """Rasterize an exact nearest-line-segment SDF from a sub-pixel contour.

    Marching squares extracts the LR coverage isocontour as a connected segment
    graph.  Output pixel centers are mapped back to LR coordinates, then exact
    point-to-segment distance is evaluated for nearby graph edges.  Lanczos
    supplies the sign, so this geometry refines known contours without creating
    components that were absent from the LR coverage.

    A later mesh variant may triangulate closed regions for an independent sign
    and hole test.  Distance should still be measured to boundary segments, not
    to triangle interiors.
    """

    alpha_array = _as_alpha_array(alpha)
    if not isinstance(scale, int) or scale < 1:
        raise ValueError("scale must be a positive integer")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between zero and one")
    if feather_width_hr <= 0:
        raise ValueError("feather_width_hr must be positive")
    if nearest_segments < 1:
        raise ValueError("nearest_segments must be positive")

    base = resize_alpha_lanczos(alpha_array, scale=scale)
    graph = _extract_segment_graph(alpha_array, level=level)
    if graph is None:
        return base
    query = _hr_queries(base.shape, scale)
    _, _, distance = _query_segment_graph(
        graph,
        query,
        nearest_segments=nearest_segments,
    )
    distance_lr = distance.reshape(base.shape)

    sign = np.where(base >= 0.5, 1.0, -1.0)
    signed_distance_hr = sign * distance_lr * float(scale)
    return sdf_to_alpha(signed_distance_hr, feather_width=feather_width_hr)


def resize_alpha_via_segment_adaptive_lanczos(
    alpha: np.ndarray,
    scale: int = 4,
    *,
    level: float = 0.5,
    feather_width_hr: float = 1.0,
    nearest_segments: int = 12,
    support_radius_normal: float = 1.5,
    support_radius_tangent: float = 3.0,
    lanczos_lobes: int = 3,
    cross_boundary_sigma: float = 1.0,
    arc_radius: float = 6.0,
    edge_band_hr: float = 3.0,
    condition_limit: float = 8.0,
    steer_kernel: bool = True,
    cross_boundary_gate: bool = False,
    segment_sample_gate: bool = False,
    limit_residual_range: bool = True,
    residual_range_epsilon: float = 0.02,
    hard_edge_gate: bool = True,
    line_art_bypass_enabled: bool = False,
    line_art_radius_limit: float = 0.75,
    line_art_min_component: int = 3,
    line_art_neighborhood: int = 3,
    line_art_peak_min: float = 0.8,
    line_art_peak_stop: float | None = None,
    line_art_max_foreground_area: int | None = 4096,
    line_art_max_canvas_area: int | None = 16384,
    line_art_ratio_start: float = 0.05,
    line_art_ratio_stop: float = 0.15,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Segment coverage plus a segment-steered Lanczos residual.

    Segment-SDF supplies analytic boundary coverage.  The difference between
    the LR alpha samples and the corresponding LR analytic coverage is then
    reconstructed with a spatially varying Lanczos kernel whose axes follow a
    coherence-weighted contour frame.  The support radii are literal LR-pixel
    support extents; the lobe count is independent.  Cross-boundary and contour
    provenance gates are separate ablations so residual reconstruction can be
    evaluated before sample restriction is introduced.

    The second return value is a reliability map.  It is zero outside the edge
    band, on broad soft transitions, and whenever the signed Lanczos weights
    are ill-conditioned.  Callers should apply this candidate as a bounded
    residual to an isotropic Lanczos base rather than replacing the base.
    """

    alpha_array = _as_alpha_array(alpha)
    if not isinstance(scale, int) or scale < 1:
        raise ValueError("scale must be a positive integer")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between zero and one")
    if feather_width_hr <= 0:
        raise ValueError("feather_width_hr must be positive")
    if nearest_segments < 1:
        raise ValueError("nearest_segments must be positive")
    if support_radius_normal <= 0 or support_radius_tangent <= 0:
        raise ValueError("support radii must be positive")
    if support_radius_tangent < support_radius_normal:
        raise ValueError("tangent support radius must not be smaller than normal support")
    if lanczos_lobes < 1:
        raise ValueError("lanczos_lobes must be positive")
    if cross_boundary_sigma <= 0 or arc_radius <= 0:
        raise ValueError("cross_boundary_sigma and arc_radius must be positive")
    if edge_band_hr <= 0 or condition_limit < 1.0:
        raise ValueError("edge_band_hr must be positive and condition_limit must be at least one")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if residual_range_epsilon < 0:
        raise ValueError("residual_range_epsilon must be non-negative")

    base = resize_alpha_lanczos(alpha_array, scale=scale)
    graph = _extract_segment_graph(alpha_array, level=level)
    if graph is None:
        return base, np.zeros_like(base)

    hr_query = _hr_queries(base.shape, scale)
    hr_segment_indices, _, hr_distance_lr = _query_segment_graph(
        graph,
        hr_query,
        nearest_segments=nearest_segments,
    )
    sign_hr = np.where(base.ravel() >= level, 1.0, -1.0).astype(np.float32)
    signed_distance_hr_lr = sign_hr * hr_distance_lr
    hr_boundary_tangents = graph.tangents[hr_segment_indices]
    hr_boundary_normals = np.stack(
        (-hr_boundary_tangents[:, 1], hr_boundary_tangents[:, 0]), axis=1
    )
    segment_coverage = _planar_segment_coverage(
        signed_distance_hr_lr,
        hr_boundary_normals,
        pixel_size_lr=1.0 / float(scale),
        feather_scale=feather_width_hr,
    ).reshape(base.shape)

    lr_height, lr_width = alpha_array.shape
    lr_y, lr_x = np.mgrid[0:lr_height, 0:lr_width]
    lr_query = np.stack([lr_y.ravel(), lr_x.ravel()], axis=1).astype(np.float32)
    lr_segment_indices, _, lr_distance = _query_segment_graph(
        graph,
        lr_query,
        nearest_segments=nearest_segments,
    )
    sign_lr = np.where(alpha_array.ravel() >= level, 1.0, -1.0).astype(np.float32)
    signed_distance_lr = sign_lr * lr_distance
    lr_boundary_tangents = graph.tangents[lr_segment_indices]
    lr_boundary_normals = np.stack(
        (-lr_boundary_tangents[:, 1], lr_boundary_tangents[:, 0]), axis=1
    )
    analytic_lr = _planar_segment_coverage(
        signed_distance_lr,
        lr_boundary_normals,
        pixel_size_lr=1.0,
    ).reshape(alpha_array.shape)
    residual_lr = (alpha_array - analytic_lr).astype(np.float32)

    if hard_edge_gate:
        gradient_x = sobel(alpha_array, axis=1, mode="nearest") / 8.0
        gradient_y = sobel(alpha_array, axis=0, mode="nearest") / 8.0
        gradient = np.hypot(gradient_x, gradient_y).astype(np.float32)
        local_kernel = np.full((5, 5), 1.0 / 25.0, dtype=np.float32)
        local_background = convolve(
            (alpha_array <= 0.05).astype(np.float32), local_kernel, mode="nearest"
        )
        local_foreground = convolve(
            (alpha_array >= 0.95).astype(np.float32), local_kernel, mode="nearest"
        )
        segment_midpoints = (graph.starts + graph.ends) * 0.5
        segment_gradient = _bilinear_sample(gradient, segment_midpoints)
        segment_binary_support = np.minimum(
            _bilinear_sample(local_background, segment_midpoints),
            _bilinear_sample(local_foreground, segment_midpoints),
        )
        gradient_score = np.clip((segment_gradient - 0.08) / 0.25, 0.0, 1.0)
        support_score = np.clip(segment_binary_support / 0.12, 0.0, 1.0)
        segment_reliability = (gradient_score * support_score).astype(np.float32)
        if line_art_bypass_enabled:
            line_art_gate_lr = persistent_centerline_support_gate(
                alpha_array,
                np.ones_like(alpha_array),
                medial_radius_limit=line_art_radius_limit,
                min_component=line_art_min_component,
                neighborhood=line_art_neighborhood,
                peak_min=line_art_peak_min,
                peak_stop=line_art_peak_stop,
                max_foreground_area=line_art_max_foreground_area,
                max_canvas_area=line_art_max_canvas_area,
                global_ratio_start=line_art_ratio_start,
                global_ratio_stop=line_art_ratio_stop,
            )
            segment_reliability = np.maximum(
                segment_reliability,
                _bilinear_sample(line_art_gate_lr, segment_midpoints),
            ).astype(np.float32)
    else:
        segment_reliability = np.ones(len(graph.starts), dtype=np.float32)

    base_edge = 4.0 * base.ravel() * (1.0 - base.ravel())
    frame_reliability = segment_reliability[hr_segment_indices]
    active = np.flatnonzero(
        (hr_distance_lr * float(scale) <= edge_band_hr)
        & (base_edge > 1e-4)
        & (frame_reliability > 1e-4)
    )
    if len(active) == 0:
        return base, np.zeros_like(base)

    support = int(np.ceil(max(support_radius_normal, support_radius_tangent)))
    offset_y, offset_x = np.mgrid[-support : support + 1, -support : support + 1]
    offset_y = offset_y.ravel().astype(np.int64)
    offset_x = offset_x.ravel().astype(np.int64)

    candidate_flat = base.ravel().copy()
    reliability_flat = np.zeros(base.size, dtype=np.float32)
    residual_flat = residual_lr.ravel()
    lr_segment_flat = lr_segment_indices
    signed_lr_flat = signed_distance_lr
    segment_flat = segment_coverage.ravel()

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
        weights = _lanczos_weight(
            float(lanczos_lobes) * normal_coordinate / float(support_radius_normal),
            lanczos_lobes,
        ) * _lanczos_weight(
            float(lanczos_lobes) * tangent_coordinate / tangent_support,
            lanczos_lobes,
        )

        if cross_boundary_gate:
            target_signed_distance = (sign_hr[target_indices] * hr_distance_lr[target_indices])[
                :, None
            ]
            sample_signed_distance = signed_lr_flat[sample_flat]
            cross_gate = np.exp(
                -np.maximum(0.0, -target_signed_distance * sample_signed_distance)
                / float(cross_boundary_sigma**2)
            ).astype(np.float32)
            weights *= cross_gate

        if segment_sample_gate:
            sample_segments = lr_segment_flat[sample_flat]
            target_contours = graph.contour_ids[target_segments][:, None]
            sample_contours = graph.contour_ids[sample_segments]
            same_contour = sample_contours == target_contours
            arc_delta = np.abs(
                graph.arc_midpoints[sample_segments] - graph.arc_midpoints[target_segments][:, None]
            )
            contour_lengths = graph.contour_lengths[target_contours]
            closed = graph.contour_closed[target_contours]
            arc_delta = np.where(
                closed, np.minimum(arc_delta, contour_lengths - arc_delta), arc_delta
            )
            arc_gate = np.exp(-0.5 * (arc_delta / float(arc_radius)) ** 2).astype(np.float32)
            weights *= np.where(same_contour, arc_gate, 0.0)

        weights *= valid
        signed_sum = np.sum(weights, axis=1)
        absolute_sum = np.sum(np.abs(weights), axis=1)
        condition = absolute_sum / np.maximum(np.abs(signed_sum), 1e-8)
        stable = (np.abs(signed_sum) > 1e-5) & (condition <= condition_limit)
        reconstructed_residual = np.zeros(len(target_indices), dtype=np.float32)
        reconstructed_residual[stable] = (
            np.sum(weights[stable] * residual_flat[sample_flat[stable]], axis=1)
            / signed_sum[stable]
        )
        if limit_residual_range:
            members = valid & (np.abs(weights) > 1e-8)
            sampled_residual = residual_flat[sample_flat]
            local_minimum = np.min(np.where(members, sampled_residual, np.inf), axis=1)
            local_maximum = np.max(np.where(members, sampled_residual, -np.inf), axis=1)
            reconstructed_residual[stable] = np.clip(
                reconstructed_residual[stable],
                local_minimum[stable] - float(residual_range_epsilon),
                local_maximum[stable] + float(residual_range_epsilon),
            )
        candidate_flat[target_indices[stable]] = np.clip(
            segment_flat[target_indices[stable]] + reconstructed_residual[stable],
            0.0,
            1.0,
        )
        reliability_flat[target_indices[stable]] = frame_reliability[target_indices[stable]]

    return (
        candidate_flat.reshape(base.shape).astype(np.float32, copy=False),
        reliability_flat.reshape(base.shape).astype(np.float32, copy=False),
    )


def resize_alpha_via_scs(
    alpha: np.ndarray,
    scale: int = 4,
    *,
    level: float = 0.5,
    feather_width_hr: float = 1.0,
    nearest_segments: int = 12,
    medial_radius_limit: float = 1.5,
    nearest_centers: int = 16,
    centerline_gate_threshold: float | None = None,
) -> np.ndarray:
    """Add a thin-branch medial-axis lower bound to Segment-SDF coverage.

    ``SCS`` means skeleton-constrained Segment-SDF.  Segment-SDF remains the
    primary boundary representation.  LR skeleton points carry a corrected
    maximal-inscribed radius, yielding the union-of-disks field
    ``max(radius - distance)``.  Only skeleton points on thin branches are
    used, and the medial field may raise coverage only inside the Lanczos 0.5
    mask.  It therefore protects weak stroke interiors without inventing an
    exterior component or rounding a Segment-SDF corner.
    """

    alpha_array = _as_alpha_array(alpha)
    if medial_radius_limit <= 0:
        raise ValueError("medial_radius_limit must be positive")
    if nearest_centers < 1:
        raise ValueError("nearest_centers must be positive")
    if centerline_gate_threshold is not None and not 0.5 < centerline_gate_threshold <= 1.0:
        raise ValueError("centerline_gate_threshold must be in (0.5, 1]")

    base = resize_alpha_lanczos(alpha_array, scale=scale)
    segment = resize_alpha_via_segments(
        alpha_array,
        scale=scale,
        level=level,
        feather_width_hr=feather_width_hr,
        nearest_segments=nearest_segments,
    )
    mask = alpha_array >= level
    if not mask.any() or mask.all():
        return segment
    skeleton = skeletonize(mask)
    if not skeleton.any():
        return segment

    # EDT measures center-to-background-center distance.  Subtracting half a
    # LR pixel converts this to an approximate center-to-cell-boundary radius.
    radius_map = np.maximum(distance_transform_edt(mask).astype(np.float32) - 0.5, 0.5)
    thin_skeleton = skeleton & (radius_map <= medial_radius_limit)
    centers = np.argwhere(thin_skeleton).astype(np.float32)
    if len(centers) == 0:
        return segment
    radii = radius_map[thin_skeleton].astype(np.float32)

    activations: np.ndarray | None = None
    if centerline_gate_threshold is not None:
        center_y_hr = np.clip(
            np.rint((centers[:, 0] + 0.5) * scale - 0.5).astype(np.int64),
            0,
            segment.shape[0] - 1,
        )
        center_x_hr = np.clip(
            np.rint((centers[:, 1] + 0.5) * scale - 0.5).astype(np.int64),
            0,
            segment.shape[1] - 1,
        )
        center_alpha = segment[center_y_hr, center_x_hr]
        activations = np.clip(
            (centerline_gate_threshold - center_alpha) / (centerline_gate_threshold - 0.5),
            0.0,
            1.0,
        ).astype(np.float32)
        active = activations > 0.0
        centers = centers[active]
        radii = radii[active]
        activations = activations[active]
        if len(centers) == 0:
            return segment
    tree = cKDTree(centers)

    output_height, output_width = base.shape
    yy, xx = np.mgrid[0:output_height, 0:output_width]
    query = np.stack(
        [
            (yy.ravel() + 0.5) / float(scale) - 0.5,
            (xx.ravel() + 0.5) / float(scale) - 0.5,
        ],
        axis=1,
    ).astype(np.float32)
    k = min(nearest_centers, len(centers))
    _, indices = tree.query(query, k=k, workers=-1)
    if k == 1:
        indices = indices[:, None]
    candidate_centers = centers[indices]
    candidate_radii = radii[indices]
    distances = np.linalg.norm(query[:, None, :] - candidate_centers, axis=2)
    inside = base >= 0.5
    candidate_sdf_hr = (candidate_radii - distances) * float(scale)
    candidate_coverage = sdf_to_alpha(candidate_sdf_hr, feather_width=feather_width_hr)
    if activations is None:
        medial_coverage = np.max(candidate_coverage, axis=1).reshape(base.shape)
        constrained = np.where(inside, np.maximum(segment, medial_coverage), segment)
    else:
        candidate_activation = activations[indices]
        segment_flat = segment.ravel()[:, None]
        lift = np.max(
            candidate_activation * np.maximum(candidate_coverage - segment_flat, 0.0),
            axis=1,
        ).reshape(base.shape)
        constrained = np.where(inside, segment + lift, segment)
    return np.clip(constrained, 0.0, 1.0).astype(np.float32)


def apply_conditional_scs_lift(
    alpha: np.ndarray,
    protected_hr: np.ndarray,
    scale: int = 4,
    *,
    medial_radius_limit: float = 0.75,
    nearest_centers: int = 16,
    centerline_gate_threshold: float = 1.0,
    min_centerline_component: int = 3,
    max_foreground_area: int | None = 4096,
    max_lift: float = 0.2,
    require_hard_foreground: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Add only a deficit-gated medial lift after protected SAL output.

    Unlike ``resize_alpha_via_scs``, this function does not reintroduce the
    Segment-SDF boundary candidate.  It constructs the same thin-skeleton
    union-of-disks lower bound, compares it directly with the already protected
    SAL result, and adds positive lift only inside the existing 0.5 topology.
    Broad translucent-only mattes are rejected unless explicitly allowed.
    """

    alpha_array = _as_alpha_array(alpha)
    protected = _as_alpha_array(protected_hr)
    if not isinstance(scale, int) or scale < 1:
        raise ValueError("scale must be a positive integer")
    expected_shape = (alpha_array.shape[0] * scale, alpha_array.shape[1] * scale)
    if protected.shape != expected_shape:
        raise ValueError(f"protected_hr must have shape {expected_shape}, got {protected.shape}")
    if medial_radius_limit <= 0 or nearest_centers < 1:
        raise ValueError("medial radius limit and nearest center count must be positive")
    if min_centerline_component < 1:
        raise ValueError("minimum centerline component size must be positive")
    if max_foreground_area is not None and max_foreground_area < 1:
        raise ValueError("maximum SCS foreground area must be positive")
    if not 0.5 < centerline_gate_threshold <= 1.0:
        raise ValueError("centerline gate threshold must be in (0.5, 1]")
    if not 0.0 <= max_lift <= 1.0:
        raise ValueError("max_lift must be in [0, 1]")

    zero_lift = np.zeros_like(protected)
    mask = alpha_array >= 0.5
    if not mask.any() or mask.all():
        return protected.copy(), zero_lift
    if max_foreground_area is not None and int(np.sum(mask)) > int(max_foreground_area):
        return protected.copy(), zero_lift
    if require_hard_foreground and (
        not np.any(alpha_array >= 0.95) or not np.any(alpha_array <= 0.05)
    ):
        return protected.copy(), zero_lift
    skeleton = skeletonize(mask)
    if not skeleton.any():
        return protected.copy(), zero_lift

    radius_map = np.maximum(distance_transform_edt(mask).astype(np.float32) - 0.5, 0.5)
    thin_skeleton = skeleton & (radius_map <= float(medial_radius_limit))
    if min_centerline_component > 1:
        components, component_count = label(
            thin_skeleton,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        if component_count == 0:
            return protected.copy(), zero_lift
        component_sizes = np.bincount(components.ravel())
        keep = component_sizes >= int(min_centerline_component)
        keep[0] = False
        thin_skeleton &= keep[components]
    centers = np.argwhere(thin_skeleton).astype(np.float32)
    if len(centers) == 0:
        return protected.copy(), zero_lift
    radii = radius_map[thin_skeleton].astype(np.float32)

    # Preserve the original SCS contract: a center is eligible only when the
    # analytic Segment-SDF representation itself has a centerline deficit.
    # Measuring the deficit on the softer SAL output activates ordinary shape
    # tips and natural silhouettes far too often.
    segment_reference = resize_alpha_via_segments(alpha_array, scale=scale)
    center_y_hr = np.clip(
        np.rint((centers[:, 0] + 0.5) * scale - 0.5).astype(np.int64),
        0,
        segment_reference.shape[0] - 1,
    )
    center_x_hr = np.clip(
        np.rint((centers[:, 1] + 0.5) * scale - 0.5).astype(np.int64),
        0,
        segment_reference.shape[1] - 1,
    )
    center_alpha = segment_reference[center_y_hr, center_x_hr]
    activations = np.clip(
        (float(centerline_gate_threshold) - center_alpha) / float(centerline_gate_threshold - 0.5),
        0.0,
        1.0,
    ).astype(np.float32)
    active = activations > 0.0
    centers = centers[active]
    radii = radii[active]
    activations = activations[active]
    if len(centers) == 0:
        return protected.copy(), zero_lift

    tree = cKDTree(centers)
    yy, xx = np.mgrid[0 : protected.shape[0], 0 : protected.shape[1]]
    query = np.stack(
        [
            (yy.ravel() + 0.5) / float(scale) - 0.5,
            (xx.ravel() + 0.5) / float(scale) - 0.5,
        ],
        axis=1,
    ).astype(np.float32)
    k = min(nearest_centers, len(centers))
    _, indices = tree.query(query, k=k, workers=-1)
    if k == 1:
        indices = indices[:, None]
    candidate_centers = centers[indices]
    candidate_radii = radii[indices]
    distances = np.linalg.norm(query[:, None, :] - candidate_centers, axis=2)
    candidate_sdf_hr = (candidate_radii - distances) * float(scale)
    candidate_coverage = sdf_to_alpha(candidate_sdf_hr, feather_width=1.0)
    reference = protected.ravel()[:, None]
    candidate_lift = activations[indices] * np.maximum(candidate_coverage - reference, 0.0)
    lift = np.max(candidate_lift, axis=1).reshape(protected.shape)
    lift = np.minimum(lift, float(max_lift)).astype(np.float32)
    lift = np.where(protected >= 0.5, lift, 0.0).astype(np.float32)
    output = np.clip(protected + lift, 0.0, 1.0).astype(np.float32)
    output = _topology_lock(protected, output)
    return output, lift


def _topology_lock(base: np.ndarray, corrected: np.ndarray) -> np.ndarray:
    """Prevent the residual from changing the base 0.5 classification."""

    inside = base >= 0.5
    below_half = np.nextafter(np.float32(0.5), np.float32(0.0))
    locked = np.where(inside, np.maximum(corrected, 0.5), np.minimum(corrected, below_half))
    return np.clip(locked, 0.0, 1.0).astype(np.float32)


def alpha_geometry_candidate(
    alpha: np.ndarray,
    scale: int = 4,
    *,
    geometry: str = "sdf",
    circle_radius: float = 2.5,
    feather_width_hr: float = 1.0,
    medial_radius_limit: float = 1.5,
    medial_nearest_centers: int = 16,
    centerline_gate_threshold: float | None = None,
    sal_support_radius_normal: float = 1.5,
    sal_support_radius_tangent: float = 3.0,
    sal_cross_boundary_sigma: float = 1.0,
    sal_arc_radius: float = 6.0,
    sal_condition_limit: float = 8.0,
    sal_steer_kernel: bool = True,
    sal_cross_boundary_gate: bool = False,
    sal_segment_sample_gate: bool = False,
    sal_limit_residual_range: bool = True,
    sal_hard_edge_gate: bool = True,
    sal_line_art_bypass_enabled: bool = False,
    sal_line_art_radius_limit: float = 0.75,
    sal_line_art_min_component: int = 3,
    sal_line_art_neighborhood: int = 3,
    sal_line_art_peak_min: float = 0.8,
    sal_line_art_peak_stop: float | None = None,
    sal_line_art_max_foreground_area: int | None = 4096,
    sal_line_art_max_canvas_area: int | None = 16384,
    sal_line_art_ratio_start: float = 0.05,
    sal_line_art_ratio_stop: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Lanczos base, geometry coverage, and geometry reliability."""

    alpha_array = _as_alpha_array(alpha)
    if geometry not in {
        "sdf",
        "circle_newton",
        "segment_sdf",
        "scs_sdf",
        "segment_adaptive_lanczos",
    }:
        raise ValueError(f"unknown geometry: {geometry}")

    base = resize_alpha_lanczos(alpha_array, scale=scale)
    if geometry == "sdf":
        candidate = resize_alpha_via_sdf(
            alpha_array,
            scale=scale,
            feather_width_hr=feather_width_hr,
        )
        reliability = np.ones_like(base)
    elif geometry == "circle_newton":
        sdf_lr, coherence_lr = circle_newton_sdf(alpha_array, radius=circle_radius)
        height, width = alpha_array.shape
        output_size = (width * scale, height * scale)
        sdf_hr = _resize_float_plane(sdf_lr, output_size) * float(scale)
        candidate = sdf_to_alpha(sdf_hr, feather_width=feather_width_hr)
        reliability = np.clip(_resize_float_plane(coherence_lr, output_size), 0.0, 1.0)
    elif geometry == "segment_sdf":
        candidate = resize_alpha_via_segments(
            alpha_array,
            scale=scale,
            feather_width_hr=feather_width_hr,
        )
        reliability = np.ones_like(base)
    elif geometry == "scs_sdf":
        candidate = resize_alpha_via_scs(
            alpha_array,
            scale=scale,
            feather_width_hr=feather_width_hr,
            medial_radius_limit=medial_radius_limit,
            nearest_centers=medial_nearest_centers,
            centerline_gate_threshold=centerline_gate_threshold,
        )
        reliability = np.ones_like(base)
    else:
        candidate, reliability = resize_alpha_via_segment_adaptive_lanczos(
            alpha_array,
            scale=scale,
            feather_width_hr=feather_width_hr,
            support_radius_normal=sal_support_radius_normal,
            support_radius_tangent=sal_support_radius_tangent,
            cross_boundary_sigma=sal_cross_boundary_sigma,
            arc_radius=sal_arc_radius,
            condition_limit=sal_condition_limit,
            steer_kernel=sal_steer_kernel,
            cross_boundary_gate=sal_cross_boundary_gate,
            segment_sample_gate=sal_segment_sample_gate,
            limit_residual_range=sal_limit_residual_range,
            hard_edge_gate=sal_hard_edge_gate,
            line_art_bypass_enabled=sal_line_art_bypass_enabled,
            line_art_radius_limit=sal_line_art_radius_limit,
            line_art_min_component=sal_line_art_min_component,
            line_art_neighborhood=sal_line_art_neighborhood,
            line_art_peak_min=sal_line_art_peak_min,
            line_art_peak_stop=sal_line_art_peak_stop,
            line_art_max_foreground_area=sal_line_art_max_foreground_area,
            line_art_max_canvas_area=sal_line_art_max_canvas_area,
            line_art_ratio_start=sal_line_art_ratio_start,
            line_art_ratio_stop=sal_line_art_ratio_stop,
        )

    return base, candidate, reliability


def apply_bounded_alpha_residual(
    base: np.ndarray,
    candidate: np.ndarray,
    reliability: np.ndarray,
    *,
    max_residual: float = 0.25,
    strength: float = 0.75,
    gate_floor: float = 0.25,
    edge_gate: bool = True,
    topology_lock: bool = True,
    additional_gate: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a bounded geometry residual to an existing coverage base.

    ``edge_gate`` retains the historical C5-A/C5-B behavior by tapering the
    residual toward the Lanczos base away from alpha=0.5.  It can be disabled
    for candidates such as SAL that already provide their own spatial
    reliability and edge-band restriction.  Residual clamping and topology
    locking remain independent controls.
    """

    base_array = _as_alpha_array(base)
    candidate_array = _as_alpha_array(candidate)
    reliability_array = _as_alpha_array(reliability)
    if candidate_array.shape != base_array.shape or reliability_array.shape != base_array.shape:
        raise ValueError("base, candidate, and reliability must have identical shapes")
    if not 0.0 <= max_residual <= 1.0:
        raise ValueError("max_residual must be in [0, 1]")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if not 0.0 <= gate_floor <= 1.0:
        raise ValueError("gate_floor must be in [0, 1]")
    if additional_gate is None:
        additional_gate_array = np.ones_like(base_array)
    else:
        additional_gate_array = _as_alpha_array(additional_gate)
        if additional_gate_array.shape != base_array.shape:
            raise ValueError("additional_gate must have the same shape as base")

    edge_weight = (
        np.clip(4.0 * base_array * (1.0 - base_array), 0.0, 1.0)
        if edge_gate
        else np.ones_like(base_array)
    )
    gated_reliability = (
        gate_floor + (1.0 - gate_floor) * reliability_array
    ) * additional_gate_array
    residual = np.clip(candidate_array - base_array, -max_residual, max_residual)
    corrected = base_array + strength * edge_weight * gated_reliability * residual
    if topology_lock:
        corrected = _topology_lock(base_array, corrected)
    return np.clip(corrected, 0.0, 1.0).astype(np.float32)


def resize_alpha_protected(
    alpha: np.ndarray,
    scale: int = 4,
    *,
    geometry: str = "sdf",
    max_residual: float = 0.25,
    strength: float = 0.75,
    gate_floor: float = 0.25,
    circle_radius: float = 2.5,
    feather_width_hr: float = 1.0,
    medial_radius_limit: float = 1.5,
    medial_nearest_centers: int = 16,
    centerline_gate_threshold: float | None = None,
    sal_support_radius_normal: float = 1.5,
    sal_support_radius_tangent: float = 3.0,
    sal_cross_boundary_sigma: float = 1.0,
    sal_arc_radius: float = 6.0,
    sal_condition_limit: float = 8.0,
    sal_steer_kernel: bool = True,
    sal_cross_boundary_gate: bool = False,
    sal_segment_sample_gate: bool = False,
    sal_limit_residual_range: bool = True,
    sal_hard_edge_gate: bool = True,
    sal_line_art_bypass_enabled: bool = False,
    sal_line_art_radius_limit: float = 0.75,
    sal_line_art_min_component: int = 3,
    sal_line_art_neighborhood: int = 3,
    sal_line_art_peak_min: float = 0.8,
    sal_line_art_peak_stop: float | None = None,
    sal_line_art_max_foreground_area: int | None = 4096,
    sal_line_art_max_canvas_area: int | None = 16384,
    sal_line_art_ratio_start: float = 0.05,
    sal_line_art_ratio_stop: float = 0.15,
    calibration_gate_enabled: bool = False,
    broad_soft_window: int = 5,
    broad_soft_start: float = 0.16,
    broad_soft_stop: float = 0.52,
    reprojection_window: int = 5,
    reprojection_margin: float = 0.0,
    reprojection_transition: float = 0.001,
    support_gate_enabled: bool = False,
    support_window: int = 9,
    support_start: float = 0.15,
    support_stop: float = 0.50,
    support_centerline_bypass_enabled: bool = False,
    support_centerline_radius_limit: float = 0.75,
    support_centerline_min_component: int = 3,
    support_centerline_neighborhood: int = 3,
    support_centerline_peak_min: float = 0.8,
    support_centerline_peak_stop: float | None = None,
    support_centerline_max_foreground_area: int | None = 4096,
    support_centerline_ratio_start: float = 0.10,
    support_centerline_ratio_stop: float = 0.30,
    scs_after_sal_enabled: bool = False,
    scs_medial_radius_limit: float = 0.75,
    scs_centerline_gate_threshold: float = 1.0,
    scs_min_centerline_component: int = 3,
    scs_max_foreground_area: int | None = 4096,
    scs_max_lift: float = 0.2,
    scs_require_hard_foreground: bool = True,
    residual_edge_gate: bool = True,
    topology_lock: bool = True,
) -> np.ndarray:
    """Lanczos coverage plus a bounded, geometry-gated alpha residual."""

    base, candidate, reliability = alpha_geometry_candidate(
        alpha,
        scale=scale,
        geometry=geometry,
        circle_radius=circle_radius,
        feather_width_hr=feather_width_hr,
        medial_radius_limit=medial_radius_limit,
        medial_nearest_centers=medial_nearest_centers,
        centerline_gate_threshold=centerline_gate_threshold,
        sal_support_radius_normal=sal_support_radius_normal,
        sal_support_radius_tangent=sal_support_radius_tangent,
        sal_cross_boundary_sigma=sal_cross_boundary_sigma,
        sal_arc_radius=sal_arc_radius,
        sal_condition_limit=sal_condition_limit,
        sal_steer_kernel=sal_steer_kernel,
        sal_cross_boundary_gate=sal_cross_boundary_gate,
        sal_segment_sample_gate=sal_segment_sample_gate,
        sal_limit_residual_range=sal_limit_residual_range,
        sal_hard_edge_gate=sal_hard_edge_gate,
        sal_line_art_bypass_enabled=sal_line_art_bypass_enabled,
        sal_line_art_radius_limit=sal_line_art_radius_limit,
        sal_line_art_min_component=sal_line_art_min_component,
        sal_line_art_neighborhood=sal_line_art_neighborhood,
        sal_line_art_peak_min=sal_line_art_peak_min,
        sal_line_art_peak_stop=sal_line_art_peak_stop,
        sal_line_art_max_foreground_area=sal_line_art_max_foreground_area,
        sal_line_art_max_canvas_area=sal_line_art_max_canvas_area,
        sal_line_art_ratio_start=sal_line_art_ratio_start,
        sal_line_art_ratio_stop=sal_line_art_ratio_stop,
    )
    calibration_gate = None
    if calibration_gate_enabled:
        ungated_proposal = apply_bounded_alpha_residual(
            base,
            candidate,
            reliability,
            max_residual=max_residual,
            strength=strength,
            gate_floor=gate_floor,
            edge_gate=residual_edge_gate,
            topology_lock=topology_lock,
        )
        calibration_gate, _ = broad_soft_reprojection_gate(
            alpha,
            base,
            ungated_proposal,
            scale=scale,
            broad_soft_window=broad_soft_window,
            broad_soft_start=broad_soft_start,
            broad_soft_stop=broad_soft_stop,
            reprojection_window=reprojection_window,
            reprojection_margin=reprojection_margin,
            reprojection_transition=reprojection_transition,
            support_gate_enabled=support_gate_enabled,
            support_window=support_window,
            support_start=support_start,
            support_stop=support_stop,
            support_centerline_bypass_enabled=support_centerline_bypass_enabled,
            support_centerline_radius_limit=support_centerline_radius_limit,
            support_centerline_min_component=support_centerline_min_component,
            support_centerline_neighborhood=support_centerline_neighborhood,
            support_centerline_peak_min=support_centerline_peak_min,
            support_centerline_peak_stop=support_centerline_peak_stop,
            support_centerline_max_foreground_area=support_centerline_max_foreground_area,
            support_centerline_ratio_start=support_centerline_ratio_start,
            support_centerline_ratio_stop=support_centerline_ratio_stop,
        )
    corrected = apply_bounded_alpha_residual(
        base,
        candidate,
        reliability,
        max_residual=max_residual,
        strength=strength,
        gate_floor=gate_floor,
        edge_gate=residual_edge_gate,
        topology_lock=topology_lock,
        additional_gate=calibration_gate,
    )
    # SCS is deliberately downstream of the fully gated SAL result.  Moving it
    # before G_support would let the calibration gate attenuate genuine medial
    # repair and would also change the segment-reference activation contract.
    if scs_after_sal_enabled:
        if geometry != "segment_adaptive_lanczos":
            raise ValueError("SCS-after-SAL requires segment_adaptive_lanczos geometry")
        corrected, _ = apply_conditional_scs_lift(
            alpha,
            corrected,
            scale=scale,
            medial_radius_limit=scs_medial_radius_limit,
            nearest_centers=medial_nearest_centers,
            centerline_gate_threshold=scs_centerline_gate_threshold,
            min_centerline_component=scs_min_centerline_component,
            max_foreground_area=scs_max_foreground_area,
            max_lift=scs_max_lift,
            require_hard_foreground=scs_require_hard_foreground,
        )
    return corrected


def main_alpha_x4_options() -> dict[str, object]:
    """Return the calibrated C5-C alpha options used by the public RGBA CLI.

    A fresh dictionary is returned so callers can record it as provenance
    without being able to mutate a module-level configuration.  These values
    are calibrated for x4 only; x2/x8 need separate area/canvas calibration.
    """

    return {
        "geometry": "segment_adaptive_lanczos",
        "max_residual": 0.2,
        "strength": 1.0,
        "gate_floor": 0.0,
        "sal_support_radius_normal": 3.0,
        "sal_support_radius_tangent": 3.0,
        "sal_steer_kernel": False,
        "sal_cross_boundary_gate": False,
        "sal_segment_sample_gate": False,
        "sal_limit_residual_range": True,
        "sal_hard_edge_gate": True,
        "sal_line_art_bypass_enabled": True,
        "sal_line_art_radius_limit": 0.75,
        "sal_line_art_min_component": 3,
        "sal_line_art_neighborhood": 3,
        "sal_line_art_peak_min": 0.8,
        "sal_line_art_peak_stop": None,
        "sal_line_art_max_foreground_area": 4096,
        "sal_line_art_max_canvas_area": 16384,
        "sal_line_art_ratio_start": 0.05,
        "sal_line_art_ratio_stop": 0.15,
        "calibration_gate_enabled": True,
        "broad_soft_window": 5,
        "broad_soft_start": 0.16,
        "broad_soft_stop": 0.52,
        "reprojection_window": 5,
        "reprojection_margin": 0.0,
        "reprojection_transition": 0.001,
        "support_gate_enabled": True,
        "support_window": 9,
        "support_start": 0.15,
        "support_stop": 0.5,
        "support_centerline_bypass_enabled": True,
        "support_centerline_radius_limit": 0.75,
        "support_centerline_min_component": 3,
        "support_centerline_neighborhood": 3,
        "support_centerline_peak_min": 0.8,
        "support_centerline_peak_stop": None,
        "support_centerline_max_foreground_area": 4096,
        "support_centerline_ratio_start": 0.1,
        "support_centerline_ratio_stop": 0.3,
        "scs_after_sal_enabled": True,
        "scs_medial_radius_limit": 0.75,
        "scs_centerline_gate_threshold": 1.0,
        "scs_min_centerline_component": 3,
        "scs_max_foreground_area": 4096,
        "scs_max_lift": 0.2,
        "scs_require_hard_foreground": True,
        "residual_edge_gate": False,
        "topology_lock": True,
    }


def resize_alpha_main_x4(alpha: np.ndarray) -> np.ndarray:
    """Upscale alpha with the selected SAL-R0 -> G_support -> SCS route."""

    return resize_alpha_protected(alpha, scale=4, **main_alpha_x4_options())


def alpha_endpoint_fraction(alpha: np.ndarray, *, endpoint: float = 0.05) -> float:
    """Measure how much LR alpha is close to exact background or foreground."""

    alpha_array = _as_alpha_array(alpha)
    if not 0.0 <= endpoint < 0.5:
        raise ValueError("endpoint must be in [0, 0.5)")
    return float(np.mean((alpha_array <= endpoint) | (alpha_array >= 1.0 - endpoint)))


def resize_alpha_ma0_x4(
    alpha: np.ndarray,
    *,
    endpoint_fraction_threshold: float = 0.30,
) -> np.ndarray:
    """Safe learned-model base: SAL for hard assets, Lanczos for broad soft alpha.

    This leaves the established public SAL route unchanged.  It is the fixed
    M-alpha-0 baseline for learned residual experiments and uses LR alpha only.
    """

    alpha_array = _as_alpha_array(alpha)
    if not 0.0 <= endpoint_fraction_threshold <= 1.0:
        raise ValueError("endpoint_fraction_threshold must be in [0, 1]")
    if alpha_endpoint_fraction(alpha_array) < endpoint_fraction_threshold:
        return resize_alpha_lanczos(alpha_array, scale=4)
    return resize_alpha_main_x4(alpha_array)


def resize_alpha_via_sdf(
    alpha: np.ndarray,
    scale: int = 4,
    *,
    threshold: float = 0.5,
    feather_width_hr: float = 1.0,
) -> np.ndarray:
    """Upscale hard alpha through an SDF and re-rasterize at HR.

    Distances are multiplied by ``scale`` after interpolation so that they stay
    expressed in output-pixel units.  This baseline cannot recover topology that
    was already lost in the LR matte.
    """

    alpha_array = _as_alpha_array(alpha)
    if not isinstance(scale, int) or scale < 1:
        raise ValueError("scale must be a positive integer")

    height, width = alpha_array.shape
    output_size = (width * scale, height * scale)
    sdf_lr = alpha_to_sdf(alpha_array, threshold=threshold)

    if np.isposinf(sdf_lr).all():
        return np.ones((height * scale, width * scale), dtype=np.float32)
    if np.isneginf(sdf_lr).all():
        return np.zeros((height * scale, width * scale), dtype=np.float32)

    sdf_hr = _resize_float_plane(sdf_lr, output_size) * float(scale)
    return sdf_to_alpha(sdf_hr, feather_width=feather_width_hr)


def load_alpha(path: str | Path) -> np.ndarray:
    """Load an alpha plane from RGBA/LA, or use a grayscale image directly."""

    image = Image.open(path)
    if image.mode in {"RGBA", "LA"}:
        plane = image.getchannel("A")
    elif image.mode in {"1", "L", "I", "F"}:
        plane = image.convert("L")
    else:
        raise ValueError(f"image has no alpha channel and is not grayscale: mode={image.mode}")
    return np.asarray(plane, dtype=np.float32) / 255.0


def save_alpha(alpha: np.ndarray, path: str | Path) -> None:
    """Save a normalized alpha plane as an 8-bit grayscale PNG."""

    alpha_array = _as_alpha_array(alpha)
    output = np.rint(alpha_array * 255.0).astype(np.uint8)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output, mode="L").save(destination)
