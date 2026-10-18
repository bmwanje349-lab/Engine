import os
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from wififix_ai.schema import encode_observation, N_FEATURES, CAUSES, ACTIONS
from wififix_ai.simulator import generate_dataset
from wififix_ai.network import WiFiFixBrain
from advisor import gate

class EngineTests(unittest.TestCase):
    def test_schema_finite_and_bounded(self):
        x = encode_observation({"connected": 1, "rssi_dbm": -55, "gateway_loss_pct": 0, "internet_loss_pct": 0, "dns_ok": 1})
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

    def test_model_save_load_roundtrip(self):
        model = WiFiFixBrain(seed=7)
        x = encode_observation({"connected": 1, "rssi_dbm": -55}).reshape(1, -1)
        c0, a0 = model.probabilities(x)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "brain.npz")
            model.save(path)
            restored = WiFiFixBrain.load(path)
            c1, a1 = restored.probabilities(x)
            self.assertTrue(np.allclose(c0, c1))
            self.assertTrue(np.allclose(a0, a1))

    def test_disconnected_gate_never_recommends_reconnect(self):
        action, _ = gate({"connected": 0, "measurement_reliability": 1},
                         "ROUTER_PATH", .99, "RECONNECT_WIFI", .99)
        self.assertEqual(action, "GATHER_MORE_EVIDENCE")

    def test_reconnect_requires_independent_evidence(self):
        obs = {"connected": 1, "measurement_reliability": .9, "rssi_dbm": -55,
               "gateway_loss_pct": 0, "dns_ok": 1, "reconnect_success_rate": .1}
        action, _ = gate(obs, "UPSTREAM_FAILURE", .9, "RECONNECT_WIFI", .95)
        self.assertEqual(action, "GATHER_MORE_EVIDENCE")

if __name__ == "__main__":
    unittest.main()
