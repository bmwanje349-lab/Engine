import math
import unittest

from train_general import _ci95, _t_critical_95


class ConfidenceIntervalTests(unittest.TestCase):
    def test_student_t_critical_values_cover_small_samples(self):
        self.assertAlmostEqual(_t_critical_95(1), 12.706, places=3)
        self.assertAlmostEqual(_t_critical_95(2), 4.303, places=3)
        self.assertAlmostEqual(_t_critical_95(29), 2.045, places=3)
        self.assertGreater(_t_critical_95(1), _t_critical_95(29))

    def test_interval_uses_paired_sample_variance(self):
        result = _ci95([1.0, 2.0, 3.0])
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["mean"], 2.0)
        self.assertLess(result["lower"], 0.0)
        self.assertGreater(result["upper"], 4.0)
        self.assertTrue(math.isfinite(result["lower"]))
        self.assertTrue(math.isfinite(result["upper"]))

    def test_empty_and_singleton_samples_are_finite(self):
        self.assertEqual(_ci95([]), {"mean": 0.0, "lower": 0.0, "upper": 0.0, "n": 0})
        single = _ci95([3.5])
        self.assertEqual(single["n"], 1)
        self.assertEqual(single["lower"], 3.5)
        self.assertEqual(single["upper"], 3.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
