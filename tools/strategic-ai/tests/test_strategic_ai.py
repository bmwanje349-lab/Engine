import json
import random
import tempfile
from collections import deque
from pathlib import Path
import unittest

from strategic_ai.gridworld import GridWorld, make_world
from strategic_ai.learning import QLearner
from strategic_ai.neural import NeuralPolicy
from strategic_ai.search import plan


class TinyDecisionTree:
    """A small known-reward problem for testing general-purpose look-ahead."""
    def actions(self, state):
        if state in ("start", "left", "right"):
            return ("LEFT", "RIGHT") if state == "start" else ("GOOD", "BAD")
        return ()

    def step(self, state, action, rng):
        if state == "start":
            nxt = "left" if action == "LEFT" else "right"
            return nxt, 0.0, False, {}
        reward = (4.0 if state == "left" and action == "GOOD" else
                  -6.0 if state == "left" else
                  1.0 if action == "GOOD" else -2.0)
        return "done", reward, True, {}

    def terminal(self, state):
        return state == "done"

    def key(self, state):
        return state

    def evaluate(self, state):
        return 0.0


class StrategicReasoningTests(unittest.TestCase):
    def test_search_finds_better_multi_step_future(self):
        result = plan(TinyDecisionTree(), "start", simulations=500, horizon=4, seed=9, top_k=2)
        self.assertEqual(result["best_action"], "LEFT")
        self.assertEqual(len(result["alternatives"]), 2)
        self.assertNotEqual(result["alternatives"][0]["first_action"], result["alternatives"][1]["first_action"])
        self.assertGreater(result["alternatives"][0]["estimated_return"], result["alternatives"][1]["estimated_return"])

    def test_search_bounded_output_and_terminal_state(self):
        world = make_world(43)
        result = plan(world, world.initial_state(), simulations=40, horizon=9, seed=1, top_k=3)
        self.assertGreaterEqual(len(result["alternatives"]), 1)
        self.assertLessEqual(len(result["alternatives"]), 3)
        self.assertEqual(result["root_visits"], 40)
        self.assertTrue(all(0 <= p["visit_share"] <= 1 for p in result["alternatives"]))
        terminal_world = GridWorld(start=(5, 5), key=(5, 5), goal=(5, 5))
        terminal = terminal_world.initial_state()
        self.assertIsNone(plan(terminal_world, terminal)["best_action"])

    def test_q_learning_saves_loads_and_uses_finite_values(self):
        learner = QLearner(seed=2)
        world = make_world(17)
        for _ in range(80):
            learner.train_episode(world)
        self.assertGreater(learner.updates, 0)
        self.assertTrue(all(v == v and abs(v) < 1e6 for row in learner.q.values() for v in row.values()))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.json"
            learner.save(path)
            restored = QLearner.load(path)
            self.assertEqual(restored.updates, learner.updates)
            self.assertEqual(restored.q, learner.q)
            self.assertGreater(path.stat().st_size, 100)

    def test_neural_distillation_checkpoint_and_bounded_prior(self):
        learner = QLearner(seed=12)
        world = make_world(33)
        for _ in range(100):
            learner.train_episode(world)
        neural, metrics = NeuralPolicy.distill(learner.q, epochs=4, seed=91)
        self.assertGreater(metrics["parameters"], 1000)
        values = neural.predict_features((1, 2, -1, -2, 0, 4, 2, 0, 1, 0, 0))
        self.assertEqual(len(values), 5)
        self.assertTrue(all(abs(v) < 1e6 for v in values))
        self.assertGreaterEqual(neural.prior(world, world.initial_state(), "UP"), -2)
        self.assertLessEqual(neural.prior(world, world.initial_state(), "UP"), 2)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "neural.json"
            neural.save(path)
            restored = NeuralPolicy.load(path)
            self.assertEqual(values, restored.predict_features((1, 2, -1, -2, 0, 4, 2, 0, 1, 0, 0)))
            self.assertGreater(path.stat().st_size, 100)

    def test_bad_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            path.write_text(json.dumps({"format": "wrong", "states": {}}))
            with self.assertRaises(ValueError):
                QLearner.load(path)

    def test_world_is_seed_reproducible(self):
        a, b = make_world(123), make_world(123)
        self.assertEqual(a.walls, b.walls)
        self.assertEqual(a.hazards, b.hazards)
        self.assertEqual(a.initial_state(), b.initial_state())

    def test_generated_world_preserves_paths_to_key_and_goal(self):
        world = make_world(981)

        def reachable(start, target):
            todo, seen = deque([start]), {start}
            while todo:
                x, y = todo.popleft()
                if (x, y) == target:
                    return True
                for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                    nxt = (x + dx, y + dy)
                    if (0 <= nxt[0] < world.width and 0 <= nxt[1] < world.height
                            and nxt not in world.walls and nxt not in seen):
                        seen.add(nxt)
                        todo.append(nxt)
            return False

        self.assertTrue(reachable(world.start, world.key_pos))
        self.assertTrue(reachable(world.key_pos, world.goal))

    def test_seeded_policy_episode_is_reproducible(self):
        world = make_world(8)
        learner = QLearner(seed=1)
        first = learner.evaluate_episode(world, seed=99)
        second = learner.evaluate_episode(world, seed=99)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main(verbosity=2)
