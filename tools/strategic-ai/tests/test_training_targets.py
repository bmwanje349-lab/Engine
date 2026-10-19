import math
import unittest

from train_general import collect_search_targets


class SearchGeneratedTrainingTargetTests(unittest.TestCase):
    def test_targets_cover_all_families_and_are_finite(self):
        targets, report = collect_search_targets(
            seed=19, target_simulations=8, target_horizon=4,
            scenarios_per_family=2, states_per_scenario=2,
        )
        self.assertGreaterEqual(len(targets), 30)
        self.assertGreater(report["target_action_values"], 0)
        self.assertGreaterEqual(report["scenario_groups"], 6)
        family_ids = set()
        for key in targets:
            feature_text = key.split("|", 1)[1] if "|" in key else key
            features = [float(value) for value in feature_text.split(",")]
            self.assertEqual(len(features), 32)
            domain = features[16:20]
            self.assertAlmostEqual(sum(domain), 1.0)
            family_ids.add(max(range(4), key=lambda index: domain[index]))
        self.assertEqual(family_ids, {0, 1, 2, 3})
        for row in targets.values():
            self.assertTrue(row)
            self.assertTrue(all(math.isfinite(float(value)) for value in row.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
