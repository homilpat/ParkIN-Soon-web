"""Regression tests for validated screening thresholds and stage voting."""

import unittest
from pathlib import Path

from late_fusion import load_stage_thresholds, majority_vote_stage


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "modeling" / "fusion_config.json"


class StageThresholdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thresholds = load_stage_thresholds(CONFIG)

    def test_validated_thresholds_are_loaded_from_config(self):
        self.assertAlmostEqual(self.thresholds["olf"], 0.4488094796813394)
        self.assertAlmostEqual(self.thresholds["img"], 0.3898262083530426)
        self.assertAlmostEqual(self.thresholds["kin"], 0.5418)

    def test_two_valid_positive_votes_create_caution_stage(self):
        probabilities = {
            "olf": self.thresholds["olf"],
            "img": self.thresholds["img"],
            "kin": self.thresholds["kin"] - 0.01,
        }
        result = majority_vote_stage(probabilities, self.thresholds)
        self.assertEqual(result["stage"], "caution")
        self.assertEqual(result["positive_count"], 2)

    def test_fewer_than_two_valid_modalities_require_retest(self):
        result = majority_vote_stage(
            {"olf": 0.8, "img": 0.8, "kin": 0.8},
            self.thresholds,
            quality_scores={"olf": 1.0, "img": 0.2, "kin": 0.0},
        )
        self.assertEqual(result["stage"], "retest")
        self.assertEqual(result["valid_count"], 1)


if __name__ == "__main__":
    unittest.main()
