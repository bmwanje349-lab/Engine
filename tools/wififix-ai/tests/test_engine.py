import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from wififix_ai.schema import encode_observation, N_FEATURES, CAUSES, ACTIONS
from wififix_ai.simulator import generate_dataset
from wififix_ai.network import WiFiFixBrain, MIN_MODEL_BYTES, HIDDEN_SIZES
from wififix_ai.planner import plan_diagnostics
from advisor import gate
from train import load_verified_field_examples


class EngineTests(unittest.TestCase):
    def test_schema_finite_and_bounded(self):
        x = encode_observation({"connected": 1, "rssi_dbm": -55, "gateway_loss_pct": 0,
                                "internet_loss_pct": 0, "dns_ok": 1})
        self.assertEqual(x.shape, (N_FEATURES,))
        self.assertTrue(np.isfinite(x).all())
        self.assertGreaterEqual(float(x.min()), 0)
        self.assertLessEqual(float(x.max()), 1)

    def test_simulator_labels_and_shapes(self):
        X, yc, ya = generate_dataset(1024, 7)
        self.assertEqual(X.shape, (1024, N_FEATURES))
        self.assertGreater(len(set(yc.tolist())), 5)
        self.assertTrue(set(yc).issubset(set(range(len(CAUSES)))))
        self.assertTrue(set(ya).issubset(set(range(len(ACTIONS)))))

    def test_model_save_load_roundtrip_and_real_parameter_size(self):
        model = WiFiFixBrain(seed=7)
        self.assertEqual(HIDDEN_SIZES, (4096, 1536))
        self.assertGreaterEqual(model.parameter_count, 6_000_000)
        self.assertGreaterEqual(model.trainable_bytes, MIN_MODEL_BYTES)
        x = encode_observation({"connected": 1, "rssi_dbm": -55}).reshape(1, -1)
        c0, a0 = model.probabilities(x)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "brain.npz")
            model.save(path)
            self.assertGreaterEqual(os.path.getsize(path), MIN_MODEL_BYTES)
            restored = WiFiFixBrain.load(path)
            c1, a1 = restored.probabilities(x)
            self.assertTrue(np.allclose(c0, c1))
            self.assertTrue(np.allclose(a0, a1))

    def test_old_checkpoint_is_rejected_instead_of_silently_reused(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "old.npz")
            np.savez(path, W1=np.zeros((N_FEATURES, 128), dtype=np.float32),
                     W2=np.zeros((128, 64), dtype=np.float32))
            with self.assertRaises(ValueError):
                WiFiFixBrain.load(path)

    def test_chess_style_planner_returns_diverse_contingent_paths(self):
        prior = np.asarray([.18, .14, .14, .12, .14, .12, .10, .06], dtype=np.float64)
        observation = {"connected": 1, "rssi_dbm": -60, "measurement_reliability": .7}
        plan = plan_diagnostics(prior, observation=observation, max_depth=3, top_paths=3)
        self.assertEqual(plan["strategy"], "bounded_expectimax_with_bayesian_branch_updates")
        paths = plan["candidate_paths"]
        self.assertEqual(len(paths), 3)
        self.assertEqual(len({p["test"] for p in paths}), 3)
        for path in paths:
            self.assertEqual(len(path["branches"]), 2)
            branch_sum = sum(branch["probability"] for branch in path["branches"].values())
            self.assertAlmostEqual(branch_sum, 1.0, delta=.002)
            self.assertTrue(all("most_likely_causes" in b for b in path["branches"].values()))

    def test_disconnected_gate_never_recommends_reconnect(self):
        action, _ = gate({"connected": 0, "measurement_reliability": 1},
                         "ROUTER_PATH", .99, "RECONNECT_WIFI", .99)
        self.assertEqual(action, "GATHER_MORE_EVIDENCE")

    def test_reconnect_requires_independent_evidence(self):
        obs = {"connected": 1, "measurement_reliability": .9, "rssi_dbm": -55,
               "gateway_loss_pct": 0, "dns_ok": 1, "reconnect_success_rate": .1}
        action, _ = gate(obs, "UPSTREAM_FAILURE", .9, "RECONNECT_WIFI", .95)
        self.assertEqual(action, "GATHER_MORE_EVIDENCE")

    def test_field_training_requires_verified_privacy_scrubbed_data(self):
        row = {
            "observation": {"connected": 1, "rssi_dbm": -60, "dns_ok": 1},
            "fault_label": "HEALTHY",
            "action_label": "NO_ACTION",
            "review_status": "verified",
            "privacy_scrubbed": True,
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "field.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            X, yc, ya = load_verified_field_examples(path)
            self.assertEqual(X.shape, (1, N_FEATURES))
            self.assertEqual(CAUSES[int(yc[0])], "HEALTHY")
            self.assertEqual(ACTIONS[int(ya[0])], "NO_ACTION")
            row["review_status"] = "example_only"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                load_verified_field_examples(path)


if __name__ == "__main__":
    unittest.main()
