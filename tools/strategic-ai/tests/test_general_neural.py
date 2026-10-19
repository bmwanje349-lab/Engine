import math
import tempfile
import unittest
from pathlib import Path

from strategic_ai.benchmarks import ResourceAllocationProblem, NimProblem, encode_features
from strategic_ai.general_neural import GeneralNeuralPolicy


class GeneralNeuralDomainCoverageTests(unittest.TestCase):
    def test_model_records_training_domains_and_rejects_unseen_domain_prior(self):
        targets = {}
        for seed in range(1, 500):
            problem = ResourceAllocationProblem(seed)
            key = encode_features(problem.features(problem.initial_state()))
            targets[key] = {
                "UP": 1.0, "RIGHT": 2.0, "DOWN": 0.0, "LEFT": -1.0, "REST": -0.5,
            }
            if len(targets) >= 60:
                break
        self.assertGreaterEqual(len(targets), 40)
        model, metrics = GeneralNeuralPolicy.distill(targets, epochs=8, seed=13)
        self.assertEqual(model.trained_domains, frozenset({0}))
        self.assertTrue(math.isfinite(metrics["final_validation_mse"]))
        resource = ResourceAllocationProblem(43)
        nim = NimProblem(70)
        self.assertTrue(model.supports_features(resource.features(resource.initial_state())))
        self.assertFalse(model.supports_features(nim.features(nim.initial_state())))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "general.json"
            model.save(path)
            restored = GeneralNeuralPolicy.load(path)
            self.assertEqual(restored.trained_domains, frozenset({0}))
            self.assertFalse(restored.supports_features(nim.features(nim.initial_state())))
            self.assertEqual(
                model.predict_features(resource.features(resource.initial_state())),
                restored.predict_features(resource.features(resource.initial_state())),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
