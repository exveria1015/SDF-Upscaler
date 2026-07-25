"""Rasterized vector-like structure fields for structure-conditioned SR.

The extractor intentionally stays deterministic and model-independent.  It turns
multiscale RGB edges into local geometric coordinates that can be consumed by an
output gate today and by HAT feature modulation later.  Open contours are kept;
the implementation never fills or forcibly closes them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import (
    convolve,
    distance_transform_edt,
    gaussian_filter,
    label,
    map_coordinates,
    sobel,
)
from skimage.feature import canny
from skimage.morphology import skeletonize

STRUCTURE_FIELD_CHANNELS = (
    "edge_confidence",
    "signed_distance",
    "unsigned_distance",
    "normal_x",
    "normal_y",
    "tangent_x",
    "tangent_y",
    "curvature",
    "junction_confidence",
    "luminance_contrast",
    "chroma_contrast",
)


@dataclass(frozen=True)
class StructureField:
    """Dense geometric conditioning plus discrete contour provenance.

    ``features`` has shape ``(H, W, C)`` in the order declared by
    :data:`STRUCTURE_FIELD_CHANNELS`.  Distances are normalized by the requested
    propagation radius.  Direction, curvature, and signed contrast channels use
    ``[-1, 1]``; confidence and unsigned channels use ``[0, 1]``.
    """

    features: np.ndarray
    edge_mask: np.ndarray
    contour_labels: np.ndarray
    channel_names: tuple[str, ...] = STRUCTURE_FIELD_CHANNELS

    def channel(self, name: str) -> np.ndarray:
        """Return one named feature plane without copying it."""

        try:
            index = self.channel_names.index(name)
        except ValueError as error:
            raise KeyError(name) from error
        return self.features[..., index]


def _as_linear_rgb(rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"rgb must have shape HxWx3, got {array.shape}")
    if array.shape[0] < 3 or array.shape[1] < 3:
        raise ValueError("rgb height and width must both be at least three pixels")
    if array.dtype == np.uint8:
        srgb = array.astype(np.float32) / 255.0
    elif np.issubdtype(array.dtype, np.floating):
        srgb = array.astype(np.float32)
        if not np.isfinite(srgb).all():
            raise ValueError("rgb contains NaN or infinity")
        if float(srgb.min()) < 0.0 or float(srgb.max()) > 1.0:
            raise ValueError("floating-point rgb must be in [0, 1]")
    else:
        raise ValueError("rgb must use uint8 or floating-point values")
    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def _gradient(plane: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smoothed = gaussian_filter(plane, sigma=sigma, mode="reflect")
    gradient_x = sobel(smoothed, axis=1, mode="reflect") / 8.0
    gradient_y = sobel(smoothed, axis=0, mode="reflect") / 8.0
    magnitude = np.hypot(gradient_x, gradient_y)
    return (
        gradient_x.astype(np.float32),
        gradient_y.astype(np.float32),
        magnitude.astype(np.float32),
    )


def _sample(plane: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return map_coordinates(
        plane,
        (y, x),
        order=1,
        mode="nearest",
        prefilter=False,
    ).astype(np.float32)


def extract_structure_field(
    rgb: np.ndarray,
    *,
    scales: tuple[float, ...] = (0.8, 1.4, 2.2),
    low_threshold: float = 0.025,
    high_threshold: float = 0.075,
    min_component_size: int = 3,
    propagation_radius: float = 6.0,
    contrast_offset: float = 1.25,
) -> StructureField:
    """Extract a multiscale edge-local geometric field from RGB.

    The strongest local gradient among luminance and the three linear RGB
    channels defines an oriented normal.  Canny candidates from every channel
    are linked into an eight-connected skeleton.  Each image location then gets
    the nearest contour's local signed distance, frame, curvature, junction
    evidence, and left/right colour contrast.
    """

    linear = _as_linear_rgb(rgb)
    if not scales or any(not np.isfinite(scale) or scale <= 0.0 for scale in scales):
        raise ValueError("scales must contain positive finite values")
    if not 0.0 <= low_threshold < high_threshold <= 1.0:
        raise ValueError("Canny thresholds must satisfy 0 <= low < high <= 1")
    if min_component_size < 1:
        raise ValueError("min_component_size must be positive")
    if not np.isfinite(propagation_radius) or propagation_radius <= 0.0:
        raise ValueError("propagation_radius must be positive and finite")
    if not np.isfinite(contrast_offset) or contrast_offset <= 0.0:
        raise ValueError("contrast_offset must be positive and finite")

    luma = (0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]).astype(
        np.float32
    )
    opponent_a = (0.5 * (linear[..., 0] - linear[..., 1]) + 0.5).astype(np.float32)
    opponent_b = (0.5 * (linear[..., 2] - 0.5 * (linear[..., 0] + linear[..., 1])) + 0.5).astype(
        np.float32
    )

    source_planes = (luma, linear[..., 0], linear[..., 1], linear[..., 2])
    edge_mask = np.zeros(luma.shape, dtype=bool)
    confidence = np.zeros(luma.shape, dtype=np.float32)
    best_magnitude = np.zeros(luma.shape, dtype=np.float32)
    best_x = np.zeros(luma.shape, dtype=np.float32)
    best_y = np.zeros(luma.shape, dtype=np.float32)

    for scale in scales:
        for plane in source_planes:
            gradient_x, gradient_y, magnitude = _gradient(plane, float(scale))
            stronger = magnitude > best_magnitude
            best_magnitude[stronger] = magnitude[stronger]
            best_x[stronger] = gradient_x[stronger]
            best_y[stronger] = gradient_y[stronger]
            confidence = np.maximum(
                confidence,
                1.0 - np.exp(-magnitude / 0.06),
            ).astype(np.float32)
            edge_mask |= canny(
                plane,
                sigma=float(scale),
                low_threshold=low_threshold,
                high_threshold=high_threshold,
            )

    candidate_labels, candidate_count = label(
        edge_mask,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    if candidate_count:
        component_sizes = np.bincount(candidate_labels.ravel())
        retained = component_sizes >= min_component_size
        retained[0] = False
        edge_mask = retained[candidate_labels]
    else:
        edge_mask.fill(False)
    edge_mask = skeletonize(edge_mask).astype(bool)
    connectivity = np.ones((3, 3), dtype=np.uint8)
    contour_labels, _ = label(edge_mask, structure=connectivity)
    contour_labels = contour_labels.astype(np.int32, copy=False)

    height, width = luma.shape
    feature_shape = (height, width, len(STRUCTURE_FIELD_CHANNELS))
    if not edge_mask.any():
        return StructureField(
            features=np.zeros(feature_shape, dtype=np.float32),
            edge_mask=edge_mask,
            contour_labels=contour_labels,
        )

    normal_denominator = np.maximum(best_magnitude, 1e-8)
    normal_x_raw = (best_x / normal_denominator).astype(np.float32)
    normal_y_raw = (best_y / normal_denominator).astype(np.float32)
    distance, nearest = distance_transform_edt(~edge_mask, return_indices=True)
    nearest_y, nearest_x = nearest
    normal_x = normal_x_raw[nearest_y, nearest_x]
    normal_y = normal_y_raw[nearest_y, nearest_x]
    normal_length = np.maximum(np.hypot(normal_x, normal_y), 1e-8)
    normal_x = (normal_x / normal_length).astype(np.float32)
    normal_y = (normal_y / normal_length).astype(np.float32)
    tangent_x = (-normal_y).astype(np.float32)
    tangent_y = normal_x.astype(np.float32)

    yy, xx = np.mgrid[:height, :width]
    delta_y = yy.astype(np.float32) - nearest_y.astype(np.float32)
    delta_x = xx.astype(np.float32) - nearest_x.astype(np.float32)
    signed_distance = (delta_x * normal_x + delta_y * normal_y) / propagation_radius
    signed_distance = np.clip(signed_distance, -1.0, 1.0).astype(np.float32)
    unsigned_distance = np.clip(distance / propagation_radius, 0.0, 1.0).astype(np.float32)

    edge_confidence = confidence[nearest_y, nearest_x] * np.exp(-distance / propagation_radius)
    edge_confidence = np.clip(edge_confidence, 0.0, 1.0).astype(np.float32)

    smooth_normal_x = gaussian_filter(normal_x_raw, sigma=1.0, mode="reflect")
    smooth_normal_y = gaussian_filter(normal_y_raw, sigma=1.0, mode="reflect")
    curvature_raw = (
        sobel(smooth_normal_x, axis=1, mode="reflect")
        + sobel(smooth_normal_y, axis=0, mode="reflect")
    ) / 8.0
    curvature = np.clip(
        curvature_raw[nearest_y, nearest_x] / 0.5,
        -1.0,
        1.0,
    ).astype(np.float32)

    neighbor_count = convolve(
        edge_mask.astype(np.uint8),
        connectivity,
        mode="constant",
        cval=0,
    ) - edge_mask.astype(np.uint8)
    junction = edge_mask & (neighbor_count >= 3)
    junction_confidence = gaussian_filter(
        junction.astype(np.float32),
        sigma=1.0,
        mode="constant",
    )
    peak = float(junction_confidence.max())
    if peak > 0.0:
        junction_confidence /= peak
    junction_confidence = junction_confidence[nearest_y, nearest_x] * np.exp(
        -distance / propagation_radius
    )
    junction_confidence = np.clip(junction_confidence, 0.0, 1.0).astype(np.float32)

    edge_y = nearest_y.astype(np.float32)
    edge_x = nearest_x.astype(np.float32)
    sample_x_plus = edge_x + normal_x * contrast_offset
    sample_y_plus = edge_y + normal_y * contrast_offset
    sample_x_minus = edge_x - normal_x * contrast_offset
    sample_y_minus = edge_y - normal_y * contrast_offset
    luma_contrast = _sample(luma, sample_y_plus, sample_x_plus) - _sample(
        luma, sample_y_minus, sample_x_minus
    )
    chroma_a_delta = _sample(opponent_a, sample_y_plus, sample_x_plus) - _sample(
        opponent_a, sample_y_minus, sample_x_minus
    )
    chroma_b_delta = _sample(opponent_b, sample_y_plus, sample_x_plus) - _sample(
        opponent_b, sample_y_minus, sample_x_minus
    )
    chroma_contrast = np.hypot(chroma_a_delta, chroma_b_delta)
    support = np.exp(-distance / propagation_radius).astype(np.float32)
    luma_contrast = np.clip(luma_contrast, -1.0, 1.0).astype(np.float32) * support
    chroma_contrast = np.clip(chroma_contrast, 0.0, 1.0).astype(np.float32) * support

    features = np.stack(
        (
            edge_confidence,
            signed_distance,
            unsigned_distance,
            normal_x,
            normal_y,
            tangent_x,
            tangent_y,
            curvature,
            junction_confidence,
            luma_contrast,
            chroma_contrast,
        ),
        axis=2,
    ).astype(np.float32)
    return StructureField(
        features=features,
        edge_mask=edge_mask,
        contour_labels=contour_labels,
    )
