from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from application_state import can_export_dxf, has_successful_wcs
from configuration import (
    DEFAULT_CONFIG_PATH,
    SUPPORTED_PIXELS_PER_MM,
    load_app_config,
    validate_operational_scale,
)
from main import create_argument_parser


class ConfigurationTests(unittest.TestCase):
    def test_default_yaml_is_the_effective_source(self):
        config = load_app_config()
        self.assertEqual(config.source_path, DEFAULT_CONFIG_PATH.resolve())
        self.assertEqual(config.geometry.pixels_per_mm, SUPPORTED_PIXELS_PER_MM)
        self.assertEqual(config.operational_method, "sam2")
        self.assertEqual(config.sam2.checkpoint.name, "sam2_hiera_tiny.pt")
        self.assertEqual(config.sam2.model_config, "configs/sam2/sam2_hiera_t.yaml")
        self.assertTrue(config.sam2.multimask_output)

    def test_only_validated_scale_is_accepted(self):
        self.assertEqual(validate_operational_scale(10), 10.0)
        for invalid in (1, 9.99, 20, 50):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "únicamente"):
                    validate_operational_scale(invalid)

    def test_cli_defaults_are_loaded_from_yaml(self):
        config = load_app_config()
        args = create_argument_parser(config).parse_args([])
        self.assertEqual(args.scale, config.geometry.pixels_per_mm)
        self.assertEqual(args.sam_checkpoint, str(config.sam2.checkpoint))
        self.assertEqual(args.sam_config, config.sam2.model_config)
        self.assertEqual(args.offset, config.export.offset_mm)
        self.assertEqual(args.include_wcs_reference, config.export.include_wcs_reference)


class ApplicationStateTests(unittest.TestCase):
    def test_export_requires_processing_wcs_and_paths(self):
        success = {
            "status": "SUCCESS",
            "origin": (100.0, 100.0),
            "uX": (1.0, 0.0),
            "uY": (0.0, 1.0),
        }
        failure = {"status": "WCS_NOT_FOUND"}
        self.assertTrue(has_successful_wcs(success))
        self.assertFalse(has_successful_wcs(failure))
        self.assertTrue(
            can_export_dxf(processed=True, wcs_info=success, contours=[object()])
        )
        self.assertFalse(
            can_export_dxf(processed=False, wcs_info=success, contours=[object()])
        )
        self.assertFalse(
            can_export_dxf(processed=True, wcs_info=failure, contours=[object()])
        )
        self.assertFalse(
            can_export_dxf(processed=True, wcs_info=success, contours=[])
        )
        malformed = {
            "status": "SUCCESS",
            "origin": (0.0, 0.0),
            "uX": (1.0, 0.0),
            "uY": (1.0, 0.0),
        }
        self.assertFalse(has_successful_wcs(malformed))
        ambiguous = {"status": "WCS_AMBIGUOUS"}
        self.assertFalse(
            can_export_dxf(
                processed=True,
                wcs_info=ambiguous,
                contours=[object()],
            )
        )


if __name__ == "__main__":
    unittest.main()
