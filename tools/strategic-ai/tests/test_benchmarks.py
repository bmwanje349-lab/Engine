import random
import unittest

from strategic_ai.benchmarks import (
    FAMILIES, NimProblem, ResourceAllocationProblem, RoutePlanningProblem,
    SchedulingProblem, make_problem, nim_action_values, nim_hybrid_action_values, nim_value,
)
from strategic_ai.gridworld import ACTIONS
from strategic_ai.search import plan


class ZeroEvaluator:
    def predict_features(self, features):
        return [0.0] * len(ACTIONS)


class BenchmarkFamilyTests(unittest.TestCase):
    def test_four_families_expose_fixed_feature_vectors(self):
        problems = [
            ResourceAllocationProblem(5),
            SchedulingProblem(5),
            RoutePlanningProblem(5),
            NimProblem(7),
        ]
        expected_ids = [0, 1, 2, 3]
        for problem, domain_id in zip(problems, expected_ids):
            state = problem.initial_state()
            features = problem.features(state)
            self.assertEqual(len(features), 32)
            self.assertEqual(features[16 + domain_id], 1.0)
            self.assertEqual(sum(features[16:20]), 1.0)
            self.assertTrue(problem.actions(state))

    def test_single_agent_search_works_on_all_planning_families(self):
        for family in FAMILIES[:3]:
            with self.subTest(family=family):
                problem = make_problem(family, 901)
                state = problem.initial_state()
                result = plan(problem, state, simulations=16, horizon=7, seed=17, top_k=5)
                self.assertIn(result["best_action"], problem.actions(state))
                self.assertGreater(result["root_visits"], 0)
                self.assertTrue(all(alt["first_action"] in problem.actions(state)
                                    for alt in result["alternatives"]))

    def test_resource_allocation_terminates_with_bounded_decisions(self):
        problem = ResourceAllocationProblem(21)
        rng = random.Random(9)
        state = problem.initial_state()
        for _ in range(12):
            if problem.terminal(state):
                break
            action = rng.choice(problem.actions(state))
            state, reward, done, _info = problem.step(state, action, rng)
            self.assertTrue(-100 <= reward <= 100)
            if done:
                break
        self.assertTrue(problem.terminal(state))

    def test_schedule_dependency_rules_are_enforced(self):
        problem = SchedulingProblem(19)
        state = problem.initial_state()
        self.assertEqual(problem.actions(state)[0], "UP")
        state, _reward, _done, _info = problem.step(state, "UP", random.Random(1))
        self.assertIn("RIGHT", problem.actions(state))
        self.assertIn("DOWN", problem.actions(state))
        self.assertNotIn("LEFT", problem.actions(state))

    def test_route_has_a_safe_corridor_and_finite_rewards(self):
        problem = RoutePlanningProblem(7)
        state = problem.initial_state()
        rng = random.Random(5)
        for _ in range(24):
            if problem.terminal(state):
                break
            action = problem.actions(state)[0]
            state, reward, _done, _info = problem.step(state, action, rng)
            self.assertTrue(-100 <= reward <= 100)
        self.assertEqual(len(problem.features(state)), 32)

    def test_nim_search_targets_are_minimax_consistent(self):
        self.assertEqual(nim_value(6), -1.0)
        self.assertEqual(nim_value(7), 1.0)
        values = nim_action_values(7)
        self.assertEqual(values["UP"], 1.0)
        self.assertEqual(max(values.values()), 1.0)
        hybrid, nodes = nim_hybrid_action_values(7, ZeroEvaluator(), depth=5)
        self.assertEqual(max(hybrid.values()), hybrid["UP"])
        self.assertGreater(nodes, 0)

    def test_factory_rejects_unknown_benchmark_family(self):
        with self.assertRaises(ValueError):
            make_problem("unknown", 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
