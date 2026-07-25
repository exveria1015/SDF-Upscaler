"""SDF-aware support gating for HAT RGB in RGBA super-resolution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter

from .alpha_sdf import alpha_to_sdf


@dataclass(frozen=True)
class RGBSupportGateDiagnostics:
    """LR evidence and HR weights used by the RGB support gate."""

    texture_score_lr: np.ndarray
    texture_gate_lr: np.ndarray
    reprojection_error_fallback_lr: np.ndarray
    reprojection_error_hat_lr: np.ndarray
    reprojection_gain_lr: np.ndarray
    reprojection_gate_lr: np.ndarray
    spatial_support_score_lr: np.ndarray
    spatial_support_gate_lr: np.ndarray
    sdf_interior_gate_lr: np.ndarray
    texture_support_gate_hr: np.ndarray
    opaque_hat_gate_hr: np.ndarray
    combined_hat_gate_hr: np.ndarray


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    return (clipped * clipped * (3.0 - 2.0 * clipped)).astype(np.float32)


def _as_rgb_u8(rgb: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} must have shape HxWx3")
    if array.dtype != np.uint8:
        raise ValueError(f"{name} must use uint8 RGB values")
    return array


def _as_alpha(alpha: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(alpha, dtype=np.float32)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return np.clip(array, 0.0, 1.0)


def _resize_plane(
    plane: np.ndarray,
    size: tuple[int, int],
    resample: Image.Resampling,
) -> np.ndarray:
    return np.asarray(
        Image.fromarray(np.asarray(plane, dtype=np.float32), mode="F").resize(
            size,
            resample,
        ),
        dtype=np.float32,
    )


def _resize_rgb_lanczos(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    rgb_u8 = _as_rgb_u8(rgb, "RGB")
    return np.asarray(
        Image.fromarray(rgb_u8).resize(size, Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )


def _srgb_to_linear(value: np.ndarray) -> np.ndarray:
    return np.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def _linear_to_srgb(value: np.ndarray) -> np.ndarray:
    return np.where(
        value <= 0.0031308,
        value * 12.92,
        1.055 * np.maximum(value, 0.0) ** (1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def _downsample_rgb(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    rgb_float = _as_rgb_u8(rgb, "RGB").astype(np.float32) / 255.0
    return np.stack(
        [
            _resize_plane(rgb_float[..., channel], size, Image.Resampling.LANCZOS)
            for channel in range(3)
        ],
        axis=2,
    )


def main_rgb_support_options() -> dict[str, float | int]:
    """Return the initial conservative x4 SDF/texture support profile."""

    return {
        "hat_alpha_start": 0.995,
        "hat_alpha_stop": 1.0,
        "texture_window": 3,
        "texture_dilation": 3,
        "texture_start": 0.03,
        "texture_stop": 0.12,
        "sdf_alpha_threshold": 0.10,
        "sdf_start_lr": 0.5,
        "sdf_stop_lr": 1.5,
        "reprojection_window": 5,
        "reprojection_margin": 0.0,
        "reprojection_transition": 0.002,
        "spatial_support_window": 5,
        "spatial_support_start": 0.10,
        "spatial_support_stop": 0.40,
        "alpha_presence_start": 0.02,
        "alpha_presence_stop": 0.10,
        "support_residual_strength": 0.10,
    }


def resize_bounded_rgb_lanczos(rgb_lr: np.ndarray, scale: int = 4) -> np.ndarray:
    """Resize the exact bounded HAT input into a deterministic RGB fallback."""

    rgb = _as_rgb_u8(rgb_lr, "rgb_lr")
    if not isinstance(scale, int) or scale < 1:
        raise ValueError("scale must be a positive integer")
    return _resize_rgb_lanczos(rgb, (rgb.shape[1] * scale, rgb.shape[0] * scale))


def resize_premultiplied_rgb_lanczos(
    rgb_lr: np.ndarray,
    alpha_lr: np.ndarray,
    scale: int = 4,
) -> np.ndarray:
    """Resize visible RGB in linear-light premultiplied form.

    Fully transparent hidden RGB is intentionally excluded.  This fallback is
    compositing-correct and cannot expose the artificial color field used only
    to stabilize HAT input near transparent boundaries.
    """

    rgb = _as_rgb_u8(rgb_lr, "rgb_lr")
    alpha = _as_alpha(alpha_lr, "alpha_lr")
    if alpha.shape != rgb.shape[:2]:
        raise ValueError("alpha_lr must match rgb_lr")
    if not isinstance(scale, int) or scale < 1:
        raise ValueError("scale must be a positive integer")
    size = (rgb.shape[1] * scale, rgb.shape[0] * scale)
    linear = _srgb_to_linear(rgb.astype(np.float32) / 255.0)
    premultiplied = linear * alpha[..., None]
    output_alpha = np.clip(
        _resize_plane(alpha, size, Image.Resampling.LANCZOS),
        0.0,
        1.0,
    )
    premultiplied_hr = np.stack(
        [
            _resize_plane(premultiplied[..., channel], size, Image.Resampling.LANCZOS)
            for channel in range(3)
        ],
        axis=2,
    )
    straight = np.zeros_like(premultiplied_hr)
    visible = output_alpha > 1e-6
    straight[visible] = premultiplied_hr[visible] / output_alpha[visible, None]
    srgb = np.clip(_linear_to_srgb(np.clip(straight, 0.0, 1.0)), 0.0, 1.0)
    return np.rint(srgb * 255.0).astype(np.uint8)


def sdf_texture_support_hat_rgb(
    bounded_rgb_lr: np.ndarray,
    hat_rgb_hr: np.ndarray,
    fallback_rgb_hr: np.ndarray,
    alpha_lr: np.ndarray,
    alpha_hr: np.ndarray,
    *,
    scale: int = 4,
    hat_alpha_start: float = 0.995,
    hat_alpha_stop: float = 1.0,
    texture_window: int = 3,
    texture_dilation: int = 3,
    texture_start: float = 0.03,
    texture_stop: float = 0.12,
    sdf_alpha_threshold: float = 0.10,
    sdf_start_lr: float = 0.5,
    sdf_stop_lr: float = 1.5,
    reprojection_window: int = 5,
    reprojection_margin: float = 0.0,
    reprojection_transition: float = 0.002,
    spatial_support_window: int = 5,
    spatial_support_start: float = 0.10,
    spatial_support_stop: float = 0.40,
    alpha_presence_start: float = 0.02,
    alpha_presence_stop: float = 0.10,
    support_residual_strength: float = 0.10,
) -> tuple[np.ndarray, RGBSupportGateDiagnostics]:
    """Fuse HAT and a deterministic RGB fallback using SDF and LR evidence.

    HAT is always restored in the near-opaque core. In translucent regions it
    is admitted only where all of the following hold: the LR straight RGB has
    local texture, HAT improves LR reprojection over the fallback with spatial
    support, and the sample is sufficiently far inside the alpha support SDF.
    """

    bounded = _as_rgb_u8(bounded_rgb_lr, "bounded_rgb_lr")
    hat = _as_rgb_u8(hat_rgb_hr, "hat_rgb_hr")
    fallback = _as_rgb_u8(fallback_rgb_hr, "fallback_rgb_hr")
    alpha_low = _as_alpha(alpha_lr, "alpha_lr")
    alpha_high = _as_alpha(alpha_hr, "alpha_hr")
    if not isinstance(scale, int) or scale < 1:
        raise ValueError("scale must be a positive integer")
    expected_shape = (bounded.shape[0] * scale, bounded.shape[1] * scale)
    if hat.shape != (*expected_shape, 3) or fallback.shape != hat.shape:
        raise ValueError("HR RGB shapes must match bounded_rgb_lr scaled by scale")
    if alpha_low.shape != bounded.shape[:2] or alpha_high.shape != expected_shape:
        raise ValueError("alpha shapes must match LR and scaled HR RGB shapes")
    if texture_window < 1 or texture_dilation < 1:
        raise ValueError("texture windows must be positive")
    if reprojection_window < 1 or spatial_support_window < 1:
        raise ValueError("reprojection and spatial support windows must be positive")
    if not 0.0 <= hat_alpha_start < hat_alpha_stop <= 1.0:
        raise ValueError("HAT alpha gate requires 0 <= start < stop <= 1")
    if not 0.0 <= texture_start < texture_stop <= 1.0:
        raise ValueError("texture thresholds must satisfy 0 <= start < stop <= 1")
    if not 0.0 < sdf_alpha_threshold < 1.0:
        raise ValueError("SDF alpha threshold must be in (0, 1)")
    if not 0.0 <= sdf_start_lr < sdf_stop_lr:
        raise ValueError("SDF bounds must satisfy 0 <= start < stop")
    if reprojection_margin < 0.0 or reprojection_transition <= 0.0:
        raise ValueError("reprojection margin must be non-negative and transition positive")
    if not 0.0 <= spatial_support_start < spatial_support_stop <= 1.0:
        raise ValueError("spatial support thresholds must satisfy 0 <= start < stop <= 1")
    if not 0.0 <= alpha_presence_start < alpha_presence_stop <= 1.0:
        raise ValueError("alpha presence thresholds must satisfy 0 <= start < stop <= 1")
    if not 0.0 <= support_residual_strength <= 1.0:
        raise ValueError("support residual strength must be in [0, 1]")

    rgb_lr = bounded.astype(np.float32) / 255.0
    channel_ranges = [
        maximum_filter(rgb_lr[..., channel], size=texture_window, mode="nearest")
        - minimum_filter(rgb_lr[..., channel], size=texture_window, mode="nearest")
        for channel in range(3)
    ]
    texture_score = np.maximum.reduce(channel_ranges).astype(np.float32)
    texture_gate = _smoothstep(
        (texture_score - float(texture_start)) / float(texture_stop - texture_start)
    )
    texture_gate = maximum_filter(
        texture_gate,
        size=texture_dilation,
        mode="nearest",
    ).astype(np.float32)

    lr_size = (bounded.shape[1], bounded.shape[0])
    fallback_reprojected = _downsample_rgb(fallback, lr_size)
    hat_reprojected = _downsample_rgb(hat, lr_size)
    alpha_mass = uniform_filter(
        alpha_low,
        size=reprojection_window,
        mode="nearest",
    ).astype(np.float32)
    fallback_error = uniform_filter(
        np.mean(np.abs(fallback_reprojected - rgb_lr), axis=2) * alpha_low,
        size=reprojection_window,
        mode="nearest",
    ).astype(np.float32)
    hat_error = uniform_filter(
        np.mean(np.abs(hat_reprojected - rgb_lr), axis=2) * alpha_low,
        size=reprojection_window,
        mode="nearest",
    ).astype(np.float32)
    valid_mass = alpha_mass > 1e-6
    fallback_error[valid_mass] /= alpha_mass[valid_mass]
    hat_error[valid_mass] /= alpha_mass[valid_mass]
    fallback_error[~valid_mass] = 0.0
    hat_error[~valid_mass] = 0.0
    reprojection_gain = (fallback_error - hat_error).astype(np.float32)
    reprojection_gate = _smoothstep(
        (reprojection_gain - float(reprojection_margin)) / float(reprojection_transition)
    )
    spatial_score = uniform_filter(
        reprojection_gate,
        size=spatial_support_window,
        mode="nearest",
    ).astype(np.float32)
    spatial_gate = _smoothstep(
        (spatial_score - float(spatial_support_start))
        / float(spatial_support_stop - spatial_support_start)
    )

    sdf_lr = alpha_to_sdf(alpha_low, threshold=sdf_alpha_threshold)
    if np.isposinf(sdf_lr).all():
        sdf_gate = np.ones_like(alpha_low)
    elif np.isneginf(sdf_lr).all():
        sdf_gate = np.zeros_like(alpha_low)
    else:
        sdf_gate = _smoothstep((sdf_lr - float(sdf_start_lr)) / float(sdf_stop_lr - sdf_start_lr))

    support_lr = np.clip(
        texture_gate * reprojection_gate * spatial_gate * sdf_gate,
        0.0,
        1.0,
    ).astype(np.float32)
    output_size = (hat.shape[1], hat.shape[0])
    support_hr = np.clip(
        _resize_plane(support_lr, output_size, Image.Resampling.BILINEAR),
        0.0,
        1.0,
    )
    presence_hr = _smoothstep(
        (alpha_high - float(alpha_presence_start))
        / float(alpha_presence_stop - alpha_presence_start)
    )
    # LR cycle consistency cannot observe HAT detail in the downsampling null
    # space.  Keep the support gate useful as evidence, but bound the admitted
    # soft-alpha HAT residual so high-frequency hallucinations cannot return at
    # full amplitude.  The opaque core remains unattenuated below.
    support_hr *= presence_hr * float(support_residual_strength)

    opaque_gate = _smoothstep(
        (alpha_high - float(hat_alpha_start)) / float(hat_alpha_stop - hat_alpha_start)
    )
    combined_gate = np.maximum(opaque_gate, support_hr).astype(np.float32)
    fused = (
        fallback.astype(np.float32)
        + (hat.astype(np.float32) - fallback.astype(np.float32)) * combined_gate[..., None]
    )
    output = np.rint(np.clip(fused, 0.0, 255.0)).astype(np.uint8)
    diagnostics = RGBSupportGateDiagnostics(
        texture_score_lr=texture_score,
        texture_gate_lr=texture_gate,
        reprojection_error_fallback_lr=fallback_error,
        reprojection_error_hat_lr=hat_error,
        reprojection_gain_lr=reprojection_gain,
        reprojection_gate_lr=reprojection_gate,
        spatial_support_score_lr=spatial_score,
        spatial_support_gate_lr=spatial_gate,
        sdf_interior_gate_lr=sdf_gate,
        texture_support_gate_hr=support_hr,
        opaque_hat_gate_hr=opaque_gate,
        combined_hat_gate_hr=combined_gate,
    )
    return output, diagnostics
