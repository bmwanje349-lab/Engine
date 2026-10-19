import math
import unittest

from strategic_ai.action_encoding import RESOURCE_ACTIONS, encode_action_features
from strategic_ai.benchmarks import NimProblem, ResourceAllocationProblem, RoutePlanningProblem, encode_features, nim_action_values, nim_hybrid_action_values
from strategic_ai.general_neural import GeneralNeuralPolicy, normalized_action_prior
from strategic_ai.search import SearchTree, plan


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
