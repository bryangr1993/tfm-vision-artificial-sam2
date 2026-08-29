"""Pruebas automáticas del contrato de partición asset_identity_disjoint."""

from __future__ import annotations

import unittest

from audit_identity_disjoint_dataset import audit


class IdentityDisjointDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = audit()

    def test_global_quality_gate(self) -> None:
        failures = [
            check for check in self.report["checks"]
            if check["severity"] == "critical" and not check["passed"]
        ]
        self.assertFalse(failures, failures)

    def test_asset_identity_checks_are_explicit(self) -> None:
        checks = {check["check"]: check for check in self.report["checks"]}
        for name in (
            "asset_id_cross_split",
            "asset_hash_cross_split",
            "source_filename_cross_split",
            "image_hash_cross_split",
            "mask_hash_cross_split",
            "train_cv_groups_asset_disjoint",
        ):
            self.assertIn(name, checks)
            self.assertTrue(checks[name]["passed"], checks[name])

    def test_family_is_a_shared_stratum(self) -> None:
        checks = {check["check"]: check for check in self.report["checks"]}
        self.assertTrue(checks["semantic_family_present_in_each_split"]["passed"])
        self.assertTrue(checks["semantic_families_intentionally_not_disjoint"]["passed"])


if __name__ == "__main__":
    unittest.main()
