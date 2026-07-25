from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from sdf_upscaler.structure_fusion import edge_local_structure_fusion_x4


class StructureFusionTests(unittest.TestCase):
    def test_constant_image_is_exactly_unchanged(self) -> None:
        rgb = np.full((12, 16, 3), 90, dtype=np.uint8)
        hat = np.asarray(
            Image.fromarray(rgb).resize((64, 48), Image.Resampling.BICUBIC),
            dtype=np.uint8,
        )
        output, diagnostics = edge_local_structure_fusion_x4(rgb, hat)

        np.testing.assert_array_equal(output, hat)
        self.assertFalse(diagnostics.applied_gate_hr.any())

    def test_vector_like_field_changes_a_high_contrast_edge(self) -> None:
        rgb = np.zeros((20, 24, 3), dtype=np.uint8)
        rgb[:, 12:] = (240, 190, 40)
        hat = np.asarray(
            Image.fromarray(rgb).resize((96, 80), Image.Resampling.BICUBIC),
            dtype=np.uint8,
        )
        output, diagnostics = edge_local_structure_fusion_x4(
            rgb,
            hat,
            reprojection_gate_enabled=False,
        )

        difference = np.max(np.abs(output.astype(np.int16) - hat.astype(np.int16)), axis=2)
        self.assertGreater(int(difference.max()), 0)
        self.assertTrue(diagnostics.structure_gate_hr[:, 40:56].any())
        self.assertFalse(diagnostics.structure_gate_hr[:, :20].any())
        self.assertFalse(diagnostics.structure_gate_hr[:, 76:].any())

    def test_reprojection_gate_never_expands_the_unconstrained_change(self) -> None:
        yy, xx = np.mgrid[:24, :24]
        rgb = np.stack(
            (
                np.where(xx < yy, 230, 20),
                np.where(xx < yy, 80, 190),
                np.full_like(xx, 60),
            ),
            axis=2,
        ).astype(np.uint8)
        hat = np.asarray(
            Image.fromarray(rgb).resize((96, 96), Image.Resampling.BICUBIC),
            dtype=np.uint8,
        )
        output, diagnostics = edge_local_structure_fusion_x4(rgb, hat)
        constrained = np.abs(output.astype(np.int16) - hat.astype(np.int16))
        unconstrained = np.abs(
            diagnostics.unconstrained_rgb_hr.astype(np.int16) - hat.astype(np.int16)
        )

        self.assertLessEqual(int(constrained.max()), int(unconstrained.max()))
        self.assertLessEqual(int(np.sum(constrained)), int(np.sum(unconstrained)))


if __name__ == "__main__":
    unittest.main()
