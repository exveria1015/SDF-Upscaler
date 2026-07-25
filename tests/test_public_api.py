from __future__ import annotations

import numpy as np
import pytest

from sdf_upscaler import (
    upscale_adaptive_sdf_x4,
    upscale_alpha_sdf_x4,
    upscale_color_sdf_x4,
)


def _structured_rgba(height: int = 20, width: int = 24) -> np.ndarray:
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[3:-3, 4:-4, :3] = (72, 188, 105)
    rgba[3:-3, 4:-4, 3] = 255
    rgba[6:-6, 8:-8, :3] = (235, 92, 48)
    rgba[6:-6, 8:-8, 3] = 255
    rgba[2, 7:17, :3] = (160, 210, 80)
    rgba[2, 7:17, 3] = 128
    return rgba


def test_alpha_api_accepts_uint8_and_normalized_float() -> None:
    alpha_u8 = _structured_rgba()[..., 3]
    from_u8 = upscale_alpha_sdf_x4(alpha_u8)
    from_float = upscale_alpha_sdf_x4(alpha_u8.astype(np.float32) / 255.0)

    assert from_u8.shape == (alpha_u8.shape[0] * 4, alpha_u8.shape[1] * 4)
    assert from_u8.dtype == np.float32
    np.testing.assert_array_equal(from_u8, from_float)


def test_color_and_adaptive_output_contracts() -> None:
    rgba = _structured_rgba()
    color, color_diagnostics = upscale_color_sdf_x4(rgba)
    adaptive, adaptive_diagnostics = upscale_adaptive_sdf_x4(rgba)

    assert color.shape == adaptive.shape == (rgba.shape[0] * 4, rgba.shape[1] * 4, 4)
    assert color.dtype == adaptive.dtype == np.uint8
    assert color_diagnostics.gate_hr.shape == color.shape[:2]
    assert adaptive_diagnostics.reprojection_gate_hr.shape == adaptive.shape[:2]


def test_adaptive_preserves_supplied_alpha() -> None:
    rgba = _structured_rgba()
    alpha_hr = np.repeat(np.repeat(rgba[..., 3], 4, axis=0), 4, axis=1)
    output, _ = upscale_adaptive_sdf_x4(rgba, alpha_hr=alpha_hr.astype(np.float32) / 255.0)
    np.testing.assert_array_equal(output[..., 3], alpha_hr)


def test_adaptive_hidden_rgb_is_invariant() -> None:
    first = _structured_rgba()
    second = first.copy()
    hidden = first[..., 3] == 0
    first[hidden, :3] = (255, 0, 255)
    second[hidden, :3] = (0, 255, 230)
    first_output, _ = upscale_adaptive_sdf_x4(first)
    second_output, _ = upscale_adaptive_sdf_x4(second)
    np.testing.assert_array_equal(first_output, second_output)


def test_invalid_alpha_range_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        upscale_alpha_sdf_x4(np.asarray([[1.1]], dtype=np.float32))
