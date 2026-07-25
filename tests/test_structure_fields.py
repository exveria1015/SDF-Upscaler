from __future__ import annotations

import unittest

import numpy as np

from sdf_upscaler.structure_fields import (
    STRUCTURE_FIELD_CHANNELS,
    extract_structure_field,
)


class StructureFieldTests(unittest.TestCase):
    def test_constant_rgb_has_no_structure(self) -> None:
        rgb = np.full((32, 40, 3), 127, dtype=np.uint8)
        field = extract_structure_field(rgb)

        self.assertEqual(field.features.shape, (32, 40, len(STRUCTURE_FIELD_CHANNELS)))
        self.assertEqual(field.features.dtype, np.float32)
        self.assertFalse(field.edge_mask.any())
        self.assertFalse(field.contour_labels.any())
        np.testing.assert_array_equal(field.features, 0.0)

    def test_vertical_step_produces_oriented_signed_geometry(self) -> None:
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        rgb[:, 32:] = 255
        field = extract_structure_field(rgb)

        near_edge = field.edge_mask & (np.indices(field.edge_mask.shape)[0] > 4)
        near_edge &= np.indices(field.edge_mask.shape)[0] < 43
        self.assertGreater(int(np.sum(near_edge)), 30)
        self.assertGreater(float(np.mean(np.abs(field.channel("normal_x")[near_edge]))), 0.9)
        self.assertLess(float(np.mean(np.abs(field.channel("normal_y")[near_edge]))), 0.1)
        self.assertLess(float(field.channel("signed_distance")[24, 27]), 0.0)
        self.assertGreater(float(field.channel("signed_distance")[24, 37]), 0.0)
        self.assertGreater(float(np.mean(field.channel("luminance_contrast")[near_edge])), 0.5)

    def test_isoluminant_chroma_boundary_is_detected(self) -> None:
        rgb = np.zeros((40, 48, 3), dtype=np.uint8)
        rgb[:, :24] = (0, 130, 0)
        rgb[:, 24:] = (216, 0, 0)
        field = extract_structure_field(rgb)

        self.assertTrue(field.edge_mask[:, 20:28].any())
        self.assertGreater(float(np.max(field.channel("chroma_contrast"))), 0.1)

    def test_float_range_and_named_channel_validation(self) -> None:
        rgb = np.zeros((8, 8, 3), dtype=np.float32)
        field = extract_structure_field(rgb)
        with self.assertRaises(KeyError):
            field.channel("missing")
        rgb[0, 0, 0] = 1.1
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            extract_structure_field(rgb)


if __name__ == "__main__":
    unittest.main()
