from __future__ import annotations

from contextlib import redirect_stdout
import io
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
REPOSITORY = Path(__file__).resolve().parents[2]
RECTIFIED = REPOSITORY / "datos" / "reales" / "rectificadas"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from application_state import can_export_dxf
from detect_wcs_l import (
    MAX_RAW_ORTHOGONALITY_DOT,
    MAX_WCS_DISTANCE_FROM_TL_MM,
    detect_wcs_l,
)


NEGATIVE_IDS = ("13", "15", "17")
POSITIVE_IDS = ("20", "22", "24", "26", "28", "30", "32", "34", "36", "38")
FIXTURES_AVAILABLE = all(
    (RECTIFIED / f"rectified_{sample_id}.png").is_file()
    for sample_id in NEGATIVE_IDS + POSITIVE_IDS
)


class SyntheticWCSAmbiguityTests(unittest.TestCase):
    def test_two_equivalent_l_marks_are_rejected_as_ambiguous(self):
        image = np.full((450, 450, 3), 255, dtype=np.uint8)
        for origin_x, origin_y in ((220, 100), (100, 220)):
            cv2.line(
                image,
                (origin_x, origin_y),
                (origin_x + 150, origin_y),
                (0, 0, 0),
                6,
            )
            cv2.line(
                image,
                (origin_x, origin_y),
                (origin_x, origin_y + 150),
                (0, 0, 0),
                6,
            )
        with redirect_stdout(io.StringIO()):
            result = detect_wcs_l(image, scale=10.0, marker_margin=10.0)
        self.assertEqual(result["status"], "WCS_AMBIGUOUS")
        self.assertIsNone(result["origin"])
        self.assertFalse(
            can_export_dxf(
                processed=True,
                wcs_info=result,
                contours=[object()],
            )
        )


@unittest.skipUnless(
    FIXTURES_AVAILABLE,
    "Las capturas reales no redistribuibles no están disponibles en este entorno.",
)
class RealWCSRegressionTests(unittest.TestCase):
    @staticmethod
    def _detect(sample_id):
        image = cv2.imread(str(RECTIFIED / f"rectified_{sample_id}.png"))
        if image is None:
            raise AssertionError(f"No se pudo leer la captura {sample_id}.")
        with redirect_stdout(io.StringIO()):
            return detect_wcs_l(image, scale=10.0, marker_margin=10.0)

    def test_three_captures_without_l_mark_are_rejected(self):
        for sample_id in NEGATIVE_IDS:
            with self.subTest(sample_id=sample_id):
                result = self._detect(sample_id)
                self.assertEqual(result["status"], "WCS_NOT_FOUND")
                self.assertFalse(
                    can_export_dxf(
                        processed=True,
                        wcs_info=result,
                        contours=[object()],
                    )
                )

    def test_ten_captures_with_l_mark_remain_positive_and_unambiguous(self):
        for sample_id in POSITIVE_IDS:
            with self.subTest(sample_id=sample_id):
                result = self._detect(sample_id)
                self.assertEqual(result["status"], "SUCCESS")
                self.assertEqual(result["candidate_clusters"], 1)
                self.assertLessEqual(
                    result["raw_orthogonality_dot"], MAX_RAW_ORTHOGONALITY_DOT
                )
                self.assertLessEqual(
                    result["distance_to_tl_mm"], MAX_WCS_DISTANCE_FROM_TL_MM
                )
                self.assertTrue(
                    can_export_dxf(
                        processed=True,
                        wcs_info=result,
                        contours=[object()],
                    )
                )


if __name__ == "__main__":
    unittest.main()
