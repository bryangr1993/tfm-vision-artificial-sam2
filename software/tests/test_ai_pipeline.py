from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_pipeline import HybridAISegmentationPipeline, segment_and_extract_with_ai
from segmenters.base import SegmentationResult
from segmenters.prompts import boxes_from_mask
from segmenters.sam2_segmenter import SAM2Segmenter, SAM2UnavailableError


class FakeLocalizer:
    def segment(self, image, **kwargs):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[10:40, 10:40] = 255
        return SegmentationResult(
            mask=mask,
            method="fake_classical_localizer",
            prompt_boxes=[(10, 10, 40, 40)],
        )


class FakeSAMSegmenter:
    def segment(self, image, *, prompt_boxes):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[14:36, 14:36] = 255
        return SegmentationResult(
            mask=mask,
            method="sam2_test_double",
            prompt_boxes=prompt_boxes,
        )


class FakePredictor:
    def set_image(self, image):
        self.shape = image.shape[:2]

    def predict(self, *, box, multimask_output):
        height, width = self.shape
        x1, y1, x2, y2 = map(int, box)
        useful = np.zeros((height, width), dtype=bool)
        useful[y1:y2, x1:x2] = True
        empty = np.zeros_like(useful)
        oversized = np.ones_like(useful)
        return (
            np.stack([useful, empty, oversized]),
            np.asarray([0.95, 0.70, 0.20]),
            np.zeros(3),
        )


class AIPipelineTests(unittest.TestCase):
    def test_sam2_mask_is_the_mask_received_by_contour_extraction(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        pipeline = HybridAISegmentationPipeline(FakeSAMSegmenter(), FakeLocalizer())
        received = {}

        def extractor(mask, rectified, debug_dir, image_name):
            received["mask"] = mask
            return [np.asarray([[[14, 14]], [[35, 14]], [[35, 35]], [[14, 35]]])], {
                "toppers_detected": 1,
                "discarded_components_count": 0,
            }

        result, contours, report = segment_and_extract_with_ai(
            pipeline,
            image,
            scale=10.0,
            wcs_info=None,
            debug_dir=None,
            image_name="test",
            contour_extractor=extractor,
        )
        self.assertIs(received["mask"], result.mask)
        self.assertEqual(report["segmentation_method"], "sam2_test_double")
        self.assertEqual(report["prompt_source"], "fake_classical_localizer")
        self.assertEqual(len(contours), 1)

    def test_missing_checkpoint_does_not_fall_back_to_classical(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.pt"
            segmenter = SAM2Segmenter(missing)
            with self.assertRaises(SAM2UnavailableError):
                segmenter.segment(
                    np.zeros((32, 32, 3), dtype=np.uint8),
                    prompt_boxes=[(2, 2, 20, 20)],
                )

    def test_sam2_adapter_returns_binary_mask_from_injected_predictor(self):
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        segmenter = SAM2Segmenter(
            "not_used.pt",
            predictor=FakePredictor(),
            device="cpu",
            box_margin_fraction=0.0,
        )
        result = segmenter.segment(image, prompt_boxes=[(10, 15, 40, 50)])
        self.assertEqual(result.method, "sam2_hiera_tiny_prompted")
        self.assertEqual(set(np.unique(result.mask)), {0, 255})
        self.assertGreater(np.count_nonzero(result.mask), 0)

    def test_prompt_localizer_keeps_largest_expected_components(self):
        mask = np.zeros((120, 160), dtype=np.uint8)
        for index in range(10):
            x = 2 + index * 15
            mask[10:40, x : x + 12] = 255
        boxes = boxes_from_mask(
            mask,
            min_area_px=100,
            max_area_px=1_000,
            expected_instances=8,
        )
        self.assertEqual(len(boxes), 8)


if __name__ == "__main__":
    unittest.main()
