import random
import unittest

from strategic_ai.benchmarks import (
    FAMILIES, NimProblem, ResourceAllocationProblem, RoutePlanningProblem,
    SchedulingProblem, make_problem, nim_action_values, nim_hybrid_action_values, nim_value,
)
from strategic_ai.gridworld import ACTIONS
from strategic_ai.action_encoding import RESOURCE_ACTIONS, SCHEDULING_ACTIONS, NIM_ACTIONS
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

    def test_each_domain_exposes_semantically_distinct_actions(self):
        resource = ResourceAllocationProblem(31)
        schedule = SchedulingProblem(31)
        route = RoutePlanningProblem(31)
        nim = NimProblem(10)

        self.assertEqual(resource.actions(resource.initial_state())[0], RESOURCE_ACTIONS[0])
        self.assertNotIn("UP", resource.actions(resource.initial_state()))
        self.assertEqual(schedule.actions(schedule.initial_state())[0], SCHEDULING_ACTIONS[0])
        self.assertNotIn("UP", schedule.actions(schedule.initial_state()))
        self.assertTrue(set(route.actions(route.initial_state())).issubset(set(ACTIONS)))
        self.assertEqual(nim.actions(nim.initial_state()), NIM_ACTIONS)

    def test_harder_difficulty_levels_preserve_valid_state_and_action_schemas(self):
        for family in ("resource_allocation", "scheduling", "route_planning"):
            with self.subTest(family=family):
                base = make_problem(family, 123, difficulty=0)
                medium = make_problem(family, 123, difficulty=1)
                hard = make_problem(family, 123, difficulty=2)
                for problem in (base, medium, hard):
                    features = problem.features(problem.initial_state())
                    self.assertEqual(len(features), 32)
                    self.assertTrue(all(-1.0 <= float(value) <= 1.0 for value in features))
                    self.assertTrue(problem.actions(problem.initial_state()))
        base_route = make_problem("route_planning", 123, difficulty=0)
        hard_route = make_problem("route_planning", 123, difficulty=2)
        self.assertEqual((base_route.width, base_route.height), (5, 5))
        self.assertEqual((hard_route.width, hard_route.height), (9, 9))
        self.assertGreater(hard_route.max_steps, base_route.max_steps)
        base_resource = make_problem("resource_allocation", 123, difficulty=0)
        hard_resource = make_problem("resource_allocation", 123, difficulty=2)
        self.assertGreaterEqual(max(hard_resource.demands), max(base_resource.demands))
        base_schedule = make_problem("scheduling", 123, difficulty=0)
        hard_schedule = make_problem("scheduling", 123, difficulty=2)
        self.assertLess(max(hard_schedule.deadlines), max(base_schedule.deadlines))

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
        self.assertEqual(problem.actions(state)[0], SCHEDULING_ACTIONS[0])
        state, _reward, _done, _info = problem.step(state, SCHEDULING_ACTIONS[0], random.Random(1))
        self.assertIn(SCHEDULING_ACTIONS[1], problem.actions(state))
        self.assertIn(SCHEDULING_ACTIONS[2], problem.actions(state))
        self.assertNotIn(SCHEDULING_ACTIONS[3], problem.actions(state))

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
        self.assertEqual(values[NIM_ACTIONS[0]], 1.0)
        self.assertEqual(max(values.values()), 1.0)
        hybrid, nodes = nim_hybrid_action_values(7, ZeroEvaluator(), depth=5)
        self.assertEqual(max(hybrid.values()), hybrid[NIM_ACTIONS[0]])
        self.assertGreater(nodes, 0)

    def test_factory_rejects_unknown_benchmark_family(self):
        with self.assertRaises(ValueError):
            make_problem("unknown", 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
