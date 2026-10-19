import math
import tempfile
import unittest
from pathlib import Path

from strategic_ai.action_encoding import RESOURCE_ACTIONS, actions_for_features, encode_action_features
from strategic_ai.benchmarks import NimProblem, ResourceAllocationProblem, RoutePlanningProblem, SchedulingProblem, encode_features, nim_action_values, nim_hybrid_action_values
from strategic_ai.general_neural import GeneralNeuralPolicy, normalized_action_prior
from strategic_ai.search import SearchTree, plan
from train_general import _ci95, _t_critical_95, exact_action_values


class TinyGraph:
    def __init__(self, transposed=False):
        self.transposed = transposed

    def key(self, state):
        return state

    def terminal(self, state):
        return state in ("X", "LEFT_END", "RIGHT_END")

    def actions(self, state):
        if state == "root":
            return ("left", "right")
        return ()

    def step(self, state, action, rng):
        del rng
        if self.transposed:
            return "X", 0.0, True, {}
        return ("LEFT_END" if action == "left" else "RIGHT_END"), 0.0, True, {}

    def evaluate(self, state):
        return 0.0


class PuctAndActionValueTests(unittest.TestCase):
    def test_exact_bellman_teacher_matches_successor_values(self):
        gamma = 0.97
        for family in ("resource_allocation", "scheduling"):
            problem = (ResourceAllocationProblem(37) if family == "resource_allocation"
                       else SchedulingProblem(37, difficulty=2))
            state = problem.initial_state()
            values = exact_action_values(problem, state, gamma=gamma)
            self.assertEqual(set(values), set(problem.actions(state)))
            self.assertTrue(all(math.isfinite(value) for value in values.values()))
            best = max(values, key=values.get)
            next_state, reward, done, _info = problem.step(state, best, __import__("random").Random(0))
            future = 0.0 if done else gamma * max(exact_action_values(problem, next_state, gamma=gamma).values())
            self.assertAlmostEqual(values[best], reward + future, places=6)
    def test_confidence_intervals_use_finite_sample_student_t(self):
        critical = _t_critical_95(39)
        self.assertGreater(critical, 2.01)
        self.assertLess(critical, 2.03)
        interval = _ci95([1.0] * 40)
        self.assertEqual(interval["n"], 40)
        self.assertEqual(interval["mean"], 1.0)
        self.assertEqual(interval["lower"], 1.0)
        self.assertEqual(interval["upper"], 1.0)
        self.assertGreater(_t_critical_95(9), _t_critical_95(39))

    def test_transposition_table_shares_identical_states_and_tree_can_advance(self):
        problem = TinyGraph(transposed=True)
        tree = SearchTree(max_nodes=1000)
        result = plan(problem, "root", simulations=48, horizon=4, seed=12, tree=tree)
        self.assertIn(result["best_action"], ("left", "right"))
        self.assertIn(("left", "X"), tree.root.children)
        self.assertIn(("right", "X"), tree.root.children)
        self.assertIs(tree.root.children[("left", "X")], tree.root.children[("right", "X")])
        child = tree.root.children[("left", "X")]
        tree.advance(problem, "left", "X")
        self.assertIs(tree.root, child)
        self.assertLessEqual(len(tree.nodes), 1000)

    def test_normalized_neural_priors_keep_action_order_without_scale_dominance(self):
        scores = {"A": 1000.0, "B": 999.0, "C": -500.0}
        self.assertGreater(normalized_action_prior(scores, "A"),
                           normalized_action_prior(scores, "B"))
        self.assertLessEqual(abs(normalized_action_prior(scores, "A")), 2.0)
        self.assertEqual(normalized_action_prior({"A": 7.0, "B": 7.0}, "A"), 0.0)

    def test_final_root_choice_uses_most_visited_child(self):
        class CountProblem(TinyGraph):
            def step(self, state, action, rng):
                del rng
                if state == "root":
                    reward = 1.0 if action == "left" else 0.0
                    return ("LEFT_END" if action == "left" else "RIGHT_END"), reward, True, {}
                return state, 0.0, True, {}

        result = plan(
            CountProblem(), "root", simulations=80, horizon=4, seed=19,
            action_prior=lambda _state, action: (5.0 if action == "left" else -5.0),
            prior_weight=1.0,
        )
        visits = {row["first_action"]: row["visits"] for row in result["alternatives"]}
        self.assertEqual(result["best_action"], max(visits, key=visits.get))
        self.assertEqual(result["best_action"], "left")

    def test_puct_uses_net_priors_and_state_value_leaf_evaluator(self):
        problem = TinyGraph(transposed=False)
        tree = SearchTree()
        result = plan(
            problem, "root", simulations=80, horizon=4, seed=4, tree=tree,
            action_prior=lambda _state, action: (4.0 if action == "right" else -4.0),
            prior_weight=1.0,
            value_evaluator=lambda state: (8.0 if state == "RIGHT_END" else -2.0),
        )
        visits = {row["first_action"]: row["visits"] for row in result["alternatives"]}
        self.assertGreater(visits["right"], visits["left"])
        self.assertEqual(result["selection"], "PUCT")

    def test_hybrid_never_degrades_an_exactly_solvable_adversarial_game(self):
        class UnreliableNeural:
            def supports_features(self, features):
                return True

            def predict_features(self, features):
                return [0.0] * 5

        # The benchmark is finite take-away Nim. Exact minimax is available,
        # so a learned leaf estimate is not allowed to overturn a proven move.
        for stones in range(1, 37):
            exact = nim_action_values(stones)
            hybrid, _solved_positions = nim_hybrid_action_values(
                stones, UnreliableNeural(), depth=5
            )
            self.assertEqual(hybrid, exact, f"pile={stones}")
            self.assertIn(max(exact, key=exact.get), NimProblem(stones).actions(stones))

    def test_domain_calibration_is_selected_and_round_trips(self):
        zeros_w1 = [[0.0] * 96 for _ in range(40)]
        zeros_w2 = [[0.0] * 64 for _ in range(96)]
        q_means = (1.0, 2.0, 3.0, 4.0)
        v_means = (-1.0, -2.0, -3.0, -4.0)
        model = GeneralNeuralPolicy(
            zeros_w1, [0.0] * 96, zeros_w2, [0.0] * 64,
            [0.0] * 64, [0.0], [0.0] * 64, [0.0],
            trained_domains=(0, 1, 2, 3),
            q_mean=q_means, q_scale=(1.0, 1.0, 1.0, 1.0),
            value_mean=v_means, value_scale=(1.0, 1.0, 1.0, 1.0),
        )
        problems = (
            ResourceAllocationProblem(14), SchedulingProblem(14),
            RoutePlanningProblem(14), NimProblem(14),
        )
        for domain_id, problem in enumerate(problems):
            features = problem.features(problem.initial_state())
            action = actions_for_features(features)[0]
            self.assertEqual(
                model.predict_action_value(features, encode_action_features(features, action)),
                q_means[domain_id],
            )
            self.assertEqual(model.predict_value_features(features), v_means[domain_id])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibrated.json"
            model.save(path)
            loaded = GeneralNeuralPolicy.load(path)
            features = problems[2].features(problems[2].initial_state())
            action = actions_for_features(features)[0]
            self.assertEqual(loaded.predict_action_value(features, action), q_means[2])

    def test_action_encoder_describes_state_action_pairs(self):
        resource = ResourceAllocationProblem(12)
        route = RoutePlanningProblem(12)
        resource_features = resource.features(resource.initial_state())
        route_features = route.features(route.initial_state())
        self.assertEqual(len(encode_action_features(resource_features, RESOURCE_ACTIONS[0])), 8)
        self.assertNotEqual(
            encode_action_features(resource_features, RESOURCE_ACTIONS[0]),
            encode_action_features(resource_features, RESOURCE_ACTIONS[1]),
        )
        self.assertNotEqual(
            encode_action_features(route_features, "UP"),
            encode_action_features(route_features, "RIGHT"),
        )
        self.assertTrue(all(math.isfinite(v) for v in encode_action_features(route_features, "REST")))

    def test_validation_split_is_by_scenario_and_checkpoint_has_value_head(self):
        targets = {}
        for scenario_seed in range(1, 90):
            problem = ResourceAllocationProblem(scenario_seed)
            feature_key = encode_features(problem.features(problem.initial_state()))
            targets[f"scenario-{scenario_seed}|{feature_key}"] = {
                RESOURCE_ACTIONS[0]: 1.0 + scenario_seed % 3,
                RESOURCE_ACTIONS[1]: 2.0,
                RESOURCE_ACTIONS[2]: 0.5,
                RESOURCE_ACTIONS[3]: -0.5,
                RESOURCE_ACTIONS[4]: -1.0,
            }
            if len(targets) >= 70:
                break
        model, metrics = GeneralNeuralPolicy.distill(targets, epochs=3, seed=9)
        self.assertFalse(metrics["scenario_group_overlap"])
        self.assertGreater(metrics["train_scenarios"], 0)
        self.assertGreater(metrics["validation_scenarios"], 0)
        features = problem.features(problem.initial_state())
        self.assertTrue(math.isfinite(model.predict_value_features(features)))
        values = model.predict_features(features)
        self.assertEqual(len(values), 5)
        self.assertTrue(all(math.isfinite(value) for value in values))


if __name__ == "__main__":
    unittest.main(verbosity=2)
