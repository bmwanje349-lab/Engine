import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from wififix_ai.schema import encode_observation, N_FEATURES, CAUSES, ACTIONS, FEATURE_NAMES
from wififix_ai.simulator import generate_dataset
from wififix_ai.network import WiFiFixBrain, MIN_MODEL_BYTES, MIN_PARAMETER_COUNT, HIDDEN_SIZES
from wififix_ai.planner import plan_diagnostics
from advisor import gate, maybe_probe_dns
from train import load_verified_field_examples, acceptance_failures


class EngineTests(unittest.TestCase):
    def test_schema_finite_and_bounded(self):
        x = encode_observation({"connected": 1, "rssi_dbm": -55, "gateway_loss_pct": 0,
                                "internet_loss_pct": 0, "dns_ok": 1})
        self.assertEqual(x.shape, (N_FEATURES,))
        self.assertTrue(np.isfinite(x).all())
        self.assertGreaterEqual(float(x.min()), 0)
        self.assertLessEqual(float(x.max()), 1)

    def test_schema_preserves_zero_reconnect_failure_and_unknown_connection(self):
        reconnect_index = FEATURE_NAMES.index("last_reconnect_success")
        connected_index = FEATURE_NAMES.index("connected")
        failed = encode_observation({"last_reconnect_success": 0})
        absent = encode_observation({})
        self.assertEqual(float(failed[reconnect_index]), 0.0)
        self.assertEqual(float(absent[reconnect_index]), 0.5)
        self.assertEqual(float(absent[connected_index]), 0.5)

    def test_simulator_labels_and_shapes(self):
        X, yc, ya = generate_dataset(1024, 7)
        self.assertEqual(X.shape, (1024, N_FEATURES))
        self.assertGreater(len(set(yc.tolist())), 5)
        self.assertTrue(set(yc).issubset(set(range(len(CAUSES)))))
        self.assertTrue(set(ya).issubset(set(range(len(ACTIONS)))))

    def test_model_save_load_roundtrip_and_real_parameter_size(self):
        model = WiFiFixBrain(seed=7)
        self.assertEqual(HIDDEN_SIZES, (3072, 1664))
        self.assertGreaterEqual(model.parameter_count, MIN_PARAMETER_COUNT)
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
        self.assertEqual(plan["strategy"], "observation_aware_bounded_expectimax")
        paths = plan["candidate_paths"]
        self.assertEqual(len(paths), 3)
        self.assertEqual(len({p["test"] for p in paths}), 3)
        for path in paths:
            self.assertEqual(len(path["branches"]), 2)
            branch_sum = sum(branch["probability"] for branch in path["branches"].values())
            self.assertAlmostEqual(branch_sum, 1.0, delta=.002)
            self.assertTrue(all("most_likely_causes" in b for b in path["branches"].values()))

    def test_planner_skips_tests_already_supported_by_measurements(self):
        prior = np.asarray([.18, .14, .14, .12, .14, .12, .10, .06], dtype=np.float64)
        observation = {
            "connected": 1, "rssi_dbm": -60, "dns_ok": 0,
            "measurement_reliability": 1,
        }
        plan = plan_diagnostics(prior, observation=observation, max_depth=3, top_paths=3)
        seen = {item["test"]: item["outcome"] for item in plan["observed_tests"]}
        self.assertEqual(seen["CHECK_SIGNAL"], "signal_adequate")
        self.assertEqual(seen["CHECK_DNS"], "dns_fails")
        self.assertNotIn("CHECK_SIGNAL", plan["remaining_tests"])
        self.assertNotIn("CHECK_DNS", plan["remaining_tests"])
        self.assertTrue(all(path["test"] not in seen for path in plan["candidate_paths"]))
        self.assertIn("same readings a second time", plan["evidence_handling"])

    def test_planner_returns_no_duplicate_tests_when_all_tests_are_measured(self):
        prior = np.full(len(CAUSES), 1 / len(CAUSES), dtype=np.float64)
        observation = {
            "rssi_dbm": -80,
            "gateway_rtt_ms": 8, "gateway_loss_pct": 0,
            "internet_rtt_ms": 30, "internet_loss_pct": 0,
            "dns_ok": 1, "download_mbps": 80, "baseline_mbps": 100,
            "sample_count": 10, "gateway_rtt_trend_ms": 2,
            "internet_rtt_trend_ms": 10, "loss_trend_pct": 0,
        }
        plan = plan_diagnostics(prior, observation=observation, max_depth=3, top_paths=3)
        self.assertEqual(plan["candidate_paths"], [])
        self.assertEqual(plan["remaining_tests"], [])
        self.assertEqual(len(plan["observed_tests"]), 6)

    def test_empty_observation_neither_assumes_connected_nor_probes_dns(self):
        action, reason = gate({}, "DNS_FAILURE", .99, "CHECK_DNS", .99)
        self.assertEqual(action, "GATHER_MORE_EVIDENCE")
        self.assertIn("Connection state", reason)
        with patch("advisor.socket.create_connection") as ip_probe, patch("advisor.socket.getaddrinfo") as dns_probe:
            status = maybe_probe_dns({}, enabled=True)
        self.assertEqual(status, "skipped_no_observation")
        ip_probe.assert_not_called()
        dns_probe.assert_not_called()

    def test_dns_probe_does_not_call_dns_when_ip_connectivity_is_unconfirmed(self):
        obs = {"connected": 1}
        with patch("advisor.socket.create_connection", side_effect=OSError("offline")) as ip_probe, \
             patch("advisor.socket.getaddrinfo") as dns_probe:
            status = maybe_probe_dns(obs, enabled=True)
        self.assertEqual(status, "skipped_ip_connectivity_unconfirmed")
        self.assertNotIn("dns_ok", obs)
        ip_probe.assert_called_once()
        dns_probe.assert_not_called()

    def test_training_acceptance_does_not_fail_only_for_non_improving_epoch_loss(self):
        metrics = {
            "minimum_20_mb_met": True,
            "loss_improved": False,
            "final_synthetic_validation": {
                "cause_accuracy": .92,
                "action_accuracy": .90,
                "cause_balanced_accuracy": .88,
                "action_balanced_accuracy": .85,
            },
        }
        self.assertEqual(acceptance_failures(metrics), [])
        metrics["final_synthetic_validation"]["cause_balanced_accuracy"] = .20
        self.assertTrue(any("cause_balanced_accuracy" in failure for failure in acceptance_failures(metrics)))

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
