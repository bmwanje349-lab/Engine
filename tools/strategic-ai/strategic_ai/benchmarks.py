"""Synthetic benchmark families for testing planner transfer beyond grid worlds.

These environments are deliberately small and reproducible. Their scores measure
algorithm behavior on controlled tasks, not real-world competence.
"""
from __future__ import annotations

from functools import lru_cache
import random
from typing import Any

from .gridworld import ACTIONS, DELTAS


class ResourceAllocationProblem:
    """Allocate a finite budget to four projects with diminishing returns."""

    family = "resource_allocation"
    domain_id = 0
    max_decisions = 9

    def __init__(self, seed: int):
        rng = random.Random(seed)
        self.demands = tuple(2 + rng.randrange(3) for _ in range(4))
        self.values = tuple(1 + rng.randrange(5) for _ in range(4))
        self.budget = 6 + rng.randrange(3)
        marginals = sorted(
            max(0.0, self.values[i] * (1.0 - 0.18 * k) - 0.10)
            for i in range(4) for k in range(self.demands[i])
        )
        self.success_threshold = 0.60 * sum(marginals[-self.budget:])

    def initial_state(self):
        return ((0, 0, 0, 0), 0)

    def key(self, state):
        return state

    def terminal(self, state):
        allocated, decisions = state
        remaining = self.budget - sum(allocated)
        return (remaining <= 0 or decisions >= self.max_decisions
                or all(allocated[i] >= self.demands[i] for i in range(4)))

    def actions(self, state):
        if self.terminal(state):
            return ()
        allocated, decisions = state
        remaining = self.budget - sum(allocated)
        choices = [ACTIONS[i] for i in range(4)
                   if allocated[i] < self.demands[i] and remaining > 0]
        if decisions < self.max_decisions - 1:
            choices.append("REST")
        return tuple(choices)

    def step(self, state, action, rng):
        del rng
        if action not in self.actions(state):
            return state, -2.0, self.terminal(state), {"invalid_action": True}
        allocated, decisions = state
        allocated = list(allocated)
        reward = -0.12
        if action == "REST":
            reward -= 0.35
        else:
            i = ACTIONS.index(action)
            reward += self.values[i] * (1.0 - 0.18 * allocated[i]) - 0.10
            allocated[i] += 1
            if allocated[i] == self.demands[i]:
                reward += 0.25
        nxt = (tuple(allocated), decisions + 1)
        done = self.terminal(nxt)
        utility = sum(self.values[i] * allocated[i] * (1.0 - 0.09 * max(0, allocated[i] - 1))
                      for i in range(4))
        if done:
            reward += 1.0 if utility >= self.success_threshold else -0.5
        return nxt, reward, done, {"success": bool(done and utility >= self.success_threshold)}

    def evaluate(self, state):
        allocated, decisions = state
        remaining_budget = max(0, self.budget - sum(allocated))
        decisions_left = max(0, self.max_decisions - decisions)
        candidates = sorted(
            max(0.0, self.values[i] * (1.0 - 0.18 * allocated[i]) - 0.10)
            for i in range(4) for _ in range(max(0, self.demands[i] - allocated[i]))
        )
        return 0.30 * sum(candidates[-min(remaining_budget, decisions_left):]) if remaining_budget else 0.0

    def features(self, state):
        allocated, decisions = state
        remaining = max(0, self.budget - sum(allocated))
        needs = [(self.demands[i] - allocated[i]) / max(1, self.demands[i]) for i in range(4)]
        values = [self.values[i] / 5.0 for i in range(4)]
        allocated_ratio = [allocated[i] / max(1, self.demands[i]) for i in range(4)]
        marginal = [max(0.0, self.values[i] * (1.0 - 0.18 * allocated[i]) - 0.10) / 5.0 for i in range(4)]
        next_marginal = [max(0.0, self.values[i] * (1.0 - 0.18 * (allocated[i] + 1)) - 0.10) / 5.0 for i in range(4)]
        domain = [1.0, 0.0, 0.0, 0.0]
        context = [remaining / max(1, self.budget), decisions / self.max_decisions,
                   sum(allocated) / max(1, self.budget), min(1.0, remaining / 2.0)]
        legal = [float(self.demands[i] > allocated[i] and remaining > 0) for i in range(4)]
        return tuple(needs + values + allocated_ratio + marginal + domain + context + legal + next_marginal)


class SchedulingProblem:
    """Schedule a four-job dependency graph under deadlines and variable durations."""

    family = "scheduling"
    domain_id = 1
    max_time = 11

    def __init__(self, seed: int):
        rng = random.Random(seed)
        self.durations = tuple(1 + rng.randrange(3) for _ in range(4))
        self.priorities = tuple(1 + rng.randrange(4) for _ in range(4))
        self.deadlines = (
            3 + rng.randrange(2),
            5 + rng.randrange(2),
            5 + rng.randrange(2),
            9 + rng.randrange(2),
        )
        self.dependencies = (0, 1, 1, (1 << 1) | (1 << 2))

    def initial_state(self):
        return (0, 0, False, 0, 0)  # completed mask, clock, wait-used, decisions, late-job mask

    def key(self, state):
        return state

    def terminal(self, state):
        mask, clock, _wait_used, decisions, _late_mask = state
        return mask == 0b1111 or clock >= self.max_time or decisions >= 6

    def _ready(self, mask):
        return [i for i in range(4) if not (mask & (1 << i))
                and (mask & self.dependencies[i]) == self.dependencies[i]]

    def actions(self, state):
        if self.terminal(state):
            return ()
        mask, clock, wait_used, _decisions, _late_mask = state
        choices = [ACTIONS[i] for i in self._ready(mask)]
        if not wait_used and clock < self.max_time - 1:
            choices.append("REST")
        return tuple(choices)

    def step(self, state, action, rng):
        del rng
        if action not in self.actions(state):
            return state, -2.0, self.terminal(state), {"invalid_action": True}
        mask, clock, wait_used, decisions, late_mask = state
        reward = -0.10
        completed = mask
        new_clock = clock
        new_wait_used = wait_used
        if action == "REST":
            new_clock += 1
            new_wait_used = True
            reward -= 0.65
        else:
            i = ACTIONS.index(action)
            duration = self.durations[i]
            new_clock += duration
            lateness = max(0, new_clock - self.deadlines[i])
            reward += 1.25 * self.priorities[i] - 0.18 * duration - 2.25 * lateness
            if lateness == 0:
                reward += 0.45
            completed |= 1 << i
            if lateness > 0:
                late_mask |= 1 << i
            if completed == 0b1111:
                reward += 3.0
        nxt = (completed, new_clock, new_wait_used, decisions + 1, late_mask)
        done = self.terminal(nxt)
        success = bool(done and completed == 0b1111 and new_clock <= self.max_time and late_mask == 0)
        return nxt, reward, done, {"success": success, "job_completed": action if action != "REST" else None}

    def evaluate(self, state):
        mask, clock, _wait_used, _decisions, _late_mask = state
        value = -0.30 * clock
        for i in range(4):
            if not (mask & (1 << i)):
                slack = self.deadlines[i] - clock - self.durations[i]
                value += 0.25 * self.priorities[i] + max(-1.0, min(1.0, slack * 0.12))
        return value

    def features(self, state):
        mask, clock, wait_used, decisions, late_mask = state
        unfinished = [float(not (mask & (1 << i))) for i in range(4)]
        priorities = [self.priorities[i] / 4.0 for i in range(4)]
        slack = [max(-1.0, min(1.0, (self.deadlines[i] - clock - self.durations[i]) / 6.0)) for i in range(4)]
        durations = [self.durations[i] / 3.0 for i in range(4)]
        domain = [0.0, 1.0, 0.0, 0.0]
        remaining_duration = sum(self.durations[i] for i in range(4) if not (mask & (1 << i)))
        context = [clock / self.max_time, float(wait_used), mask.bit_count() / 4.0,
                   min(1.0, remaining_duration / self.max_time)]
        ready = self._ready(mask)
        ready_flags = [float(i in ready) for i in range(4)]
        dependencies_met = [float((mask & self.dependencies[i]) == self.dependencies[i]) for i in range(4)]
        return tuple(unfinished + priorities + slack + durations + domain + context + ready_flags + dependencies_met)


class RoutePlanningProblem:
    """Weighted, hazardous grid routing with energy and terrain costs."""

    family = "route_planning"
    domain_id = 2
    width = 5
    height = 5
    max_energy = 18
    max_steps = 24

    def __init__(self, seed: int):
        rng = random.Random(seed)
        self.start, self.goal = (0, 0), (4, 4)
        corridor = ({(x, 0) for x in range(self.width)}
                    | {(4, y) for y in range(self.height)})
        candidates = [(x, y) for y in range(self.height) for x in range(self.width)
                      if (x, y) not in corridor and (x, y) not in (self.start, self.goal)]
        rng.shuffle(candidates)
        self.walls = frozenset(candidates[:3])
        self.rough = { (x, y): 1 + rng.randrange(2)
                       for y in range(self.height) for x in range(self.width)
                       if (x, y) not in self.walls }
        hazard_candidates = [p for p in candidates[3:] if p not in self.walls]
        self.hazards = frozenset(hazard_candidates[:3])

    def initial_state(self):
        return (0, 0, self.max_energy, 0)

    def key(self, state):
        return state

    def terminal(self, state):
        x, y, energy, steps = state
        return ((x, y) == self.goal or energy <= 0 or steps >= self.max_steps)

    def _neighbor(self, x, y, action):
        dx, dy = DELTAS[action]
        point = (x + dx, y + dy)
        if not (0 <= point[0] < self.width and 0 <= point[1] < self.height) or point in self.walls:
            return None
        return point

    def actions(self, state):
        if self.terminal(state):
            return ()
        x, y, energy, _steps = state
        choices = [a for a in ACTIONS[:4] if self._neighbor(x, y, a) is not None]
        if energy < self.max_energy:
            choices.append("REST")
        return tuple(choices)

    def step(self, state, action, rng):
        if action not in self.actions(state):
            return state, -2.0, self.terminal(state), {"invalid_action": True}
        x, y, energy, steps = state
        reward = -0.15
        info = {"action": action}
        if action == "REST":
            energy = min(self.max_energy, energy + 3)
            reward -= 0.80
        else:
            point = self._neighbor(x, y, action)
            cost = self.rough[point]
            x, y = point
            energy -= cost
            reward -= 0.28 * cost
            if (x, y) in self.hazards and rng.random() < 0.22:
                energy -= 2
                reward -= 1.25
                info["hazard_hit"] = True
            if (x, y) == self.goal:
                reward += 12.0
                info["goal_reached"] = True
        steps += 1
        nxt = (x, y, energy, steps)
        done = self.terminal(nxt)
        if energy <= 0 and not info.get("goal_reached"):
            reward -= 5.0
            info["failure"] = "energy_depleted"
        if steps >= self.max_steps and not info.get("goal_reached"):
            reward -= 4.0
            info["failure"] = "step_limit"
        info["success"] = bool(done and (x, y) == self.goal)
        return nxt, reward, done, info

    def evaluate(self, state):
        x, y, energy, _steps = state
        distance = abs(self.goal[0] - x) + abs(self.goal[1] - y)
        return -0.65 * distance - 0.07 * (self.max_energy - energy)

    def features(self, state):
        x, y, energy, steps = state
        actions = ACTIONS[:4]
        neighbors = [self._neighbor(x, y, action) for action in actions]
        legal = [float(point is not None) for point in neighbors]
        costs = [1.0 if point is None else self.rough[point] / 2.0 for point in neighbors]
        risks = [0.0 if point is None else float(point in self.hazards) for point in neighbors]
        distance = lambda point: abs(self.goal[0] - point[0]) + abs(self.goal[1] - point[1])
        current_distance = distance((x, y))
        next_distance = [current_distance if point is None else distance(point) for point in neighbors]
        next_distance = [max(0.0, min(1.0, d / 8.0)) for d in next_distance]
        domain = [0.0, 0.0, 1.0, 0.0]
        context = [(self.goal[0] - x) / 4.0, (self.goal[1] - y) / 4.0,
                   energy / self.max_energy, steps / self.max_steps]
        expected_cost = [min(1.0, costs[i] + 0.25 * risks[i]) for i in range(4)]
        distance_gain = [max(-1.0, min(1.0, (current_distance - d) / 4.0)) for d in [current_distance if p is None else distance(p) for p in neighbors]]
        return tuple(legal + costs + risks + next_distance + domain + context + expected_cost + distance_gain)


class NimProblem:
    """Two-player take-away game: players remove one to five stones; last move wins.

    The state is expressed from the side-to-move perspective. It is used with
    negamax/minimax targets, never with the single-agent MCTS backup.
    """

    family = "adversarial"
    domain_id = 3

    def __init__(self, stones: int):
        if not 1 <= stones <= 100:
            raise ValueError("Nim benchmark pile must be between 1 and 100")
        self.stones = int(stones)

    def initial_state(self):
        return self.stones

    def key(self, state):
        return int(state)

    def terminal(self, state):
        return int(state) <= 0

    def actions(self, state):
        return tuple(ACTIONS[i] for i in range(min(5, int(state))))

    def step(self, state, action, rng):
        del rng
        if action not in self.actions(state):
            return state, -1.0, self.terminal(state), {"invalid_action": True}
        take = ACTIONS.index(action) + 1
        nxt = int(state) - take
        return nxt, (1.0 if nxt == 0 else 0.0), nxt <= 0, {"stones_taken": take}

    def evaluate(self, state):
        return 0.0

    def features(self, state):
        stones = max(0, int(state))
        next_counts = [max(0, stones - take) for take in range(1, 5)]
        remaining = [(-1.0 if stones < take else min(1.0, (stones - take) / 100.0)) for take in range(1, 5)]
        next_residue = [((stones - take) % 6) / 5.0 if stones >= take else -1.0 for take in range(1, 5)]
        legal = [float(stones >= take) for take in range(1, 5)]
        parity = [float((stones - take) % 2) if stones >= take else -1.0 for take in range(1, 5)]
        domain = [0.0, 0.0, 0.0, 1.0]
        context = [min(1.0, stones / 100.0), (stones % 6) / 5.0,
                   (stones % 5) / 4.0, float(stones % 2)]
        taking_sizes = [take / 5.0 for take in range(1, 5)]
        residue_onehot = [float(stones % 5 == residue) for residue in range(1, 5)]
        return tuple(remaining + next_residue + legal + parity + domain + context + taking_sizes + residue_onehot)


@lru_cache(maxsize=2048)
def nim_value(stones: int) -> float:
    """Exact minimax value from the side-to-move perspective."""
    if stones <= 0:
        return -1.0
    return max(-nim_value(stones - take) for take in range(1, min(5, stones) + 1))


def nim_action_values(stones: int) -> dict[str, float]:
    """Strong-search action targets (optimal returns from the current player's view)."""
    problem = NimProblem(stones)
    return {action: -nim_value(stones - (ACTIONS.index(action) + 1))
            for action in problem.actions(stones)}


def _bounded_neural_value(stones: int, neural) -> float:
    if stones <= 0:
        return -1.0
    features = NimProblem(stones).features(stones)
    outputs = neural.predict_features(features)
    legal = NimProblem(stones).actions(stones)
    if not legal:
        return -1.0
    return max(-1.0, min(1.0, max(float(outputs[ACTIONS.index(a)]) for a in legal)))


def nim_hybrid_action_values(stones: int, neural, *, depth: int = 5) -> tuple[dict[str, float], int]:
    """Bounded adversarial look-ahead with a small neural leaf-value contribution."""
    if depth < 1:
        raise ValueError("depth must be positive")
    # Never let an evaluator trained only on other domains steer an adversarial
    # search. Fall back to exact minimax until this checkpoint has game data.
    if hasattr(neural, "supports_features") and not neural.supports_features(NimProblem(stones).features(stones)):
        return nim_action_values(stones), 0
    nodes = 0

    def negamax(n: int, remaining: int) float:
        nonlocal nodes
        nodes += 1
        if n <= 0:
            return -1.0
        if remaining <= 0:
            # Conservative blending: an unfamiliar neural evaluator cannot
            # overwhelm the simple bounded fallback on a novel game state.
            learned = _bounded_neural_value(n, neural)
            fallback = 0.25 if n % 6 else -0.25
            return max(-0.95, min(0.95, 0.20 * learned + 0.80 * fallback))
        return max(-negamax(n - take, remaining - 1)
                   for take in range(1, min(5, n) + 1))

    values = {action: -negamax(stones - (ACTIONS.index(action) + 1), depth - 1)
              for action in NimProblem(stones).actions(stones)}
    return values, nodes


def make_problem(family: str, seed: int):
    if family == "resource_allocation":
        return ResourceAllocationProblem(seed)
    if family == "scheduling":
        return SchedulingProblem(seed)
    if family == "route_planning":
        return RoutePlanningProblem(seed)
    if family == "adversarial":
        return NimProblem(1 + (abs(int(seed)) % 36))
    raise ValueError(f"unknown benchmark family: {family}")


FAMILIES = ("resource_allocation", "scheduling", "route_planning", "adversarial")


def encode_features(features) -> str:
    if len(features) != 32:
        raise ValueError("general benchmark feature vectors must have exactly 32 values")
    return ",".join(f"{float(v):.7g}" for v in features)
