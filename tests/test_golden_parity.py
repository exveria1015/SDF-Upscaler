from __future__ import annotations

import hashlib

import numpy as np

from sdf_upscaler import (
    upscale_adaptive_sdf_x4,
    upscale_alpha_sdf_x4,
    upscale_color_sdf_x4,
)


def _fixture() -> np.ndarray:
    height, width = 18, 22
    yy, xx = np.mgrid[:height, :width]
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = np.stack(
        (
            (xx * 17 + yy * 3) % 256,
            (xx * 7 + yy * 19) % 256,
            (xx * 13 + yy * 11) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    mask = ((xx - 10.5) / 7.5) ** 2 + ((yy - 8.5) / 6.0) ** 2 <= 1.0
    rgba[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
    rgba[3, 6:16, 3] = 128
    return rgba


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def test_extracted_algorithms_match_source_snapshot() -> None:
    rgba = _fixture()
    alpha = upscale_alpha_sdf_x4(rgba[..., 3])
    alpha_u8 = np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    color, color_diagnostics = upscale_color_sdf_x4(rgba)
    adaptive, adaptive_diagnostics = upscale_adaptive_sdf_x4(rgba)

    assert _digest(alpha_u8) == "777276c9bbee3a9ebd57b6ae9ae80e178280d6dc0fea6ace970347b4f95d4bd3"
    assert _digest(color) == "f00ce3c04297942c59ad8b7986cb781151a8a3a22374ce1d1484e8beee10219c"
    assert _digest(adaptive) == "289a9a0f6444883306fee5d7599ce0af3d34244780515e25409bcac91fb38074"
    assert int(color_diagnostics.active_mask_hr.sum()) == 666
    assert not adaptive_diagnostics.global_reprojection_fallback
    assert adaptive_diagnostics.lr_error_final <= adaptive_diagnostics.lr_error_base + 1e-9
