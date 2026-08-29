from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import ezdxf
import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import export_dxf
from ai_pipeline import HybridAISegmentationPipeline, segment_and_extract_with_ai
from export_dxf import (
    DxfExportError,
    DxfValidationError,
    OffsetDependencyError,
    apply_offset_shapely,
    export_to_dxf,
    generate_validation_dxf,
    simplify_and_transform_contour,
    validate_dxf_file,
)
from extract_contours import extract_contours
from segmenters.base import SegmentationResult


SUCCESS_WCS = {
    "status": "SUCCESS",
    "origin": (0.0, 0.0),
    "uX": (1.0, 0.0),
    "uY": (0.0, 1.0),
}


def rectangle(x1, y1, x2, y2):
    return np.asarray(
        [[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32
    )


class ContourHierarchyTests(unittest.TestCase):
    def test_internal_hole_is_preserved_without_counting_as_another_topper(self):
        mask = np.zeros((520, 520), dtype=np.uint8)
        cv2.rectangle(mask, (40, 40), (480, 480), 255, thickness=-1)
        cv2.circle(mask, (260, 260), 70, 0, thickness=-1)

        contours, report = extract_contours(
            mask, min_hole_area_px=100, preserve_holes=True
        )

        self.assertEqual(report["toppers_detected"], 1)
        self.assertEqual(report["outer_contours_count"], 1)
        self.assertEqual(report["holes_preserved_count"], 1)
        self.assertEqual(report["cut_paths_total"], 2)
        self.assertEqual(report["path_roles"], ["outer", "hole"])
        self.assertEqual(len(contours), 2)

    def test_operational_default_exports_only_the_external_silhouette(self):
        mask = np.zeros((520, 520), dtype=np.uint8)
        cv2.rectangle(mask, (40, 40), (480, 480), 255, thickness=-1)
        cv2.circle(mask, (260, 260), 70, 0, thickness=-1)

        contours, report = extract_contours(mask)

        self.assertEqual(report["outer_contours_count"], 1)
        self.assertEqual(report["holes_preserved_count"], 0)
        self.assertEqual(report["cut_paths_total"], 1)
        self.assertEqual(report["path_roles"], ["outer"])
        self.assertEqual(len(contours), 1)


class DxfExportTests(unittest.TestCase):
    def test_vectorization_rejects_an_unvalidated_scale(self):
        with self.assertRaisesRegex(DxfExportError, "únicamente"):
            simplify_and_transform_contour(
                rectangle(100, 100, 500, 500), SUCCESS_WCS, scale=20.0
            )

    def test_roundtrip_validates_layers_closed_paths_counts_and_bounds(self):
        contours = [rectangle(100, 100, 500, 500), rectangle(220, 220, 320, 320)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "verified.dxf"
            success = export_to_dxf(
                contours,
                SUCCESS_WCS,
                output,
                offset_mm=0.2,
                contour_roles=["outer", "hole"],
                include_wcs_reference=True,
            )
            self.assertTrue(success)
            self.assertTrue(output.is_file())

            report = validate_dxf_file(
                output,
                expected_outer_count=1,
                expected_hole_count=1,
                include_wcs_reference=True,
            )
            self.assertEqual(report.outer_paths, 1)
            self.assertEqual(report.hole_paths, 1)
            self.assertEqual(report.wcs_entities, 4)
            reopened = ezdxf.readfile(output)
            self.assertIn("TOPPERS_HOLES", {layer.dxf.name for layer in reopened.layers})
            self.assertTrue(
                all(
                    entity.closed
                    for entity in reopened.modelspace()
                    if entity.dxf.layer in {"TOPPERS_CUT", "TOPPERS_HOLES"}
                )
            )

    def test_export_is_blocked_without_wcs_and_does_not_leave_a_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "blocked.dxf"
            success = export_to_dxf(
                [rectangle(100, 100, 500, 500)],
                {"status": "WCS_NOT_FOUND"},
                output,
            )
            self.assertFalse(success)
            self.assertFalse(output.exists())

    def test_export_is_blocked_when_coordinates_exceed_physical_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outside.dxf"
            success = export_to_dxf(
                [rectangle(10_000, 100, 11_000, 500)],
                SUCCESS_WCS,
                output,
            )
            self.assertFalse(success)
            self.assertFalse(output.exists())

    def test_export_is_blocked_for_a_zero_area_path(self):
        collinear = np.asarray(
            [[[100, 100]], [[200, 200]], [[300, 300]]], dtype=np.int32
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "zero_area.dxf"
            self.assertFalse(export_to_dxf([collinear], SUCCESS_WCS, output))
            self.assertFalse(output.exists())

    def test_corrupt_file_fails_roundtrip_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            corrupt = Path(directory) / "corrupt.dxf"
            corrupt.write_text("not a dxf", encoding="utf-8")
            with self.assertRaises(DxfValidationError):
                validate_dxf_file(corrupt, expected_outer_count=1)

    def test_requested_offset_never_falls_back_when_shapely_is_missing(self):
        points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        with patch.object(export_dxf, "SHAPELY_AVAILABLE", False):
            with self.assertRaisesRegex(OffsetDependencyError, "requiere Shapely"):
                apply_offset_shapely(points, 0.2)

    @unittest.skipUnless(export_dxf.SHAPELY_AVAILABLE, "Shapely no está instalado")
    def test_offset_expands_outer_path_and_contracts_hole(self):
        points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        outer = np.asarray(apply_offset_shapely(points, 1.0, role="outer"))
        hole = np.asarray(apply_offset_shapely(points, 1.0, role="hole"))
        self.assertLess(float(np.min(outer[:, 0])), 0.0)
        self.assertGreater(float(np.max(outer[:, 0])), 10.0)
        self.assertGreater(float(np.min(hole[:, 0])), 0.0)
        self.assertLess(float(np.max(hole[:, 0])), 10.0)

    def test_calibration_pattern_is_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "wcs_validation.dxf"
            self.assertTrue(generate_validation_dxf(output, "down"))
            document = ezdxf.readfile(output)
            self.assertEqual(len(list(document.modelspace())), 5)


class HeadlessIntegrationTests(unittest.TestCase):
    def test_ai_mask_to_hierarchical_contours_to_verified_dxf(self):
        class Localizer:
            def segment(self, image, **kwargs):
                return SegmentationResult(
                    mask=np.zeros(image.shape[:2], dtype=np.uint8),
                    method="classical_prompt_test_double",
                    prompt_boxes=[(30, 30, 490, 490)],
                )

        class AISegmenter:
            def segment(self, image, *, prompt_boxes):
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.rectangle(mask, (40, 40), (480, 480), 255, thickness=-1)
                cv2.circle(mask, (260, 260), 70, 0, thickness=-1)
                return SegmentationResult(
                    mask=mask,
                    method="sam2_test_double",
                    prompt_boxes=prompt_boxes,
                )

        image = np.zeros((520, 520, 3), dtype=np.uint8)
        pipeline = HybridAISegmentationPipeline(AISegmenter(), Localizer())
        result, contours, report = segment_and_extract_with_ai(
            pipeline,
            image,
            scale=10.0,
            wcs_info=SUCCESS_WCS,
            debug_dir=None,
            image_name="headless",
        )
        self.assertEqual(result.method, "sam2_test_double")
        self.assertEqual(report["prompt_source"], "classical_prompt_test_double")
        self.assertEqual(report["path_roles"], ["outer"])

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "headless_pipeline.dxf"
            self.assertTrue(
                export_to_dxf(
                    contours,
                    SUCCESS_WCS,
                    output,
                    contour_roles=report["path_roles"],
                )
            )
            roundtrip = validate_dxf_file(output, expected_outer_count=1)
            self.assertEqual(roundtrip.outer_paths, 1)
            self.assertEqual(roundtrip.hole_paths, 0)


if __name__ == "__main__":
    unittest.main()
