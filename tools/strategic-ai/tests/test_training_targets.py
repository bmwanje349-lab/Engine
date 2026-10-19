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
        family_ids = {int(key.split(",")[4]) for key in targets}
        self.assertEqual(family_ids, {0, 1, 2, 3})
        for row in targets.values():
            self.assertTrue(row)
            self.assertTrue(all(math.isfinite(float(value)) for value in row.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
