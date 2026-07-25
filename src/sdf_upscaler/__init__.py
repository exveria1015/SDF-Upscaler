"""Public API for deterministic Alpha-, Color-, and Adaptive-SDF x4 upscaling."""

from __future__ import annotations

import numpy as np

from .adaptive_color_sdf import (
    ADAPTIVE_COLOR_SDF_PIPELINE_ID,
    AdaptiveColorSDFDiagnostics,
    adaptive_color_sdf_rgb_x4,
    main_adaptive_color_sdf_options,
    upscale_rgba_adaptive_color_sdf_x4,
)
from .alpha_sdf import (
    main_alpha_x4_options,
    resize_alpha_lanczos,
    resize_alpha_ma0_x4,
    resize_alpha_main_x4,
)
from .sal_classic import (
    SAL_CLASSIC_PIPELINE_ID,
    SegmentGuidedRGBDiagnostics,
    main_sal_classic_rgb_options,
    segment_guided_normalized_lanczos_rgb,
    upscale_rgba_sal_classic_x4,
)

__version__ = "0.1.0"


def upscale_alpha_sdf_x4(alpha_lr: np.ndarray) -> np.ndarray:
    """Upscale uint8 or normalized floating-point alpha to float32 x4 alpha."""

    alpha = np.asarray(alpha_lr)
    if alpha.ndim != 2 or alpha.size == 0:
        raise ValueError("alpha_lr must be a non-empty 2D array")
    if alpha.dtype == np.uint8:
        normalized = alpha.astype(np.float32) / 255.0
    elif np.issubdtype(alpha.dtype, np.floating):
        normalized = alpha.astype(np.float32)
    else:
        raise ValueError("alpha_lr must use uint8 or floating-point values")
    if not np.isfinite(normalized).all() or normalized.min() < 0.0 or normalized.max() > 1.0:
        raise ValueError("floating-point alpha_lr must be finite and in [0, 1]")
    return resize_alpha_main_x4(normalized)


def upscale_color_sdf_x4(
    rgba_lr: np.ndarray,
    *,
    rgb_options: dict[str, object] | None = None,
) -> tuple[np.ndarray, SegmentGuidedRGBDiagnostics]:
    """Upscale straight uint8 RGBA with the Color-SDF/SAL-Classic path."""

    return upscale_rgba_sal_classic_x4(rgba_lr, rgb_options=rgb_options)


def upscale_adaptive_sdf_x4(
    rgba_lr: np.ndarray,
    *,
    alpha_hr: np.ndarray | None = None,
    adaptive_options: dict[str, float | int] | None = None,
    color_options: dict[str, object] | None = None,
) -> tuple[np.ndarray, AdaptiveColorSDFDiagnostics]:
    """Upscale straight uint8 RGBA with Adaptive-SDF and optional supplied alpha."""

    return upscale_rgba_adaptive_color_sdf_x4(
        rgba_lr,
        alpha_hr=alpha_hr,
        options=adaptive_options,
        sal_rgb_options=color_options,
    )


__all__ = [
    "ADAPTIVE_COLOR_SDF_PIPELINE_ID",
    "AdaptiveColorSDFDiagnostics",
    "SAL_CLASSIC_PIPELINE_ID",
    "SegmentGuidedRGBDiagnostics",
    "__version__",
    "adaptive_color_sdf_rgb_x4",
    "main_adaptive_color_sdf_options",
    "main_alpha_x4_options",
    "main_sal_classic_rgb_options",
    "resize_alpha_lanczos",
    "resize_alpha_ma0_x4",
    "resize_alpha_main_x4",
    "segment_guided_normalized_lanczos_rgb",
    "upscale_adaptive_sdf_x4",
    "upscale_alpha_sdf_x4",
    "upscale_color_sdf_x4",
    "upscale_rgba_adaptive_color_sdf_x4",
    "upscale_rgba_sal_classic_x4",
]
