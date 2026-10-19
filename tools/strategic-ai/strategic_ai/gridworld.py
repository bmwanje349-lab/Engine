"""Reference stochastic planning environment with a key, hazards and energy.

It is a benchmark domain, not a claim that grid navigation alone is general
intelligence. It exists to make planning and learning improvements measurable.
"""
from __future__ import annotations
from dataclasses import dataclass
import random
from typing import Hashable, Sequence

ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "REST")
DELTAS = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}


@dataclass(frozen=True)
class State:
    x: int
    y: int
    has_key: bool
    energy: int
    steps: int


class GridWorld:
    def __init__(self, width: int = 6, height: int = 6, *, walls=(), hazards=(),
                 start=(0, 0), key=(2, 2), goal=(5, 5), max_energy: int = 16,
                 max_steps: int = 45, hazard_probability: float = 0.32):
        if width < 3 or height < 3:
            raise ValueError("world dimensions must be at least 3x3")
        if not 0 <= hazard_probability <= 1:
            raise ValueError("hazard probability must be in [0,1]")
        self.width, self.height = int(width), int(height)
        self.start, self.key_pos, self.goal = tuple(start), tuple(key), tuple(goal)
        self.walls, self.hazards = frozenset(map(tuple, walls)), frozenset(map(tuple, hazards))
        for point in (self.start, self.key_pos, self.goal, *self.walls, *self.hazards):
            if not (0 <= point[0] < width and 0 <= point[1] < height):
                raise ValueError(f"point out of bounds: {point}")
        if any(point in self.walls for point in (self.start, self.key_pos, self.goal)):
            raise ValueError("start, key and goal must not be walls")
        self.max_energy, self.max_steps = max_energy, max_steps
        self.hazard_probability = float(hazard_probability)

    def initial_state(self) -> State:
        return State(self.start[0], self.start[1], self.start == self.key_pos,
                     self.max_energy, 0)

    def key(self, state: State) -> Hashable:
        return (state.x, state.y, int(state.has_key), state.energy, state.steps)

    def actions(self, state: State) -> Sequence[str]:
        if self.terminal(state):
            return ()
        legal = list(ACTIONS)
        if state.energy >= self.max_energy:
            legal.remove("REST")
        return legal

    def terminal(self, state: State) -> bool:
        return ((state.x, state.y) == self.goal and state.has_key) or state.energy <= 0 or state.steps >= self.max_steps

    def evaluate(self, state: State) -> float:
        """Conservative nonterminal leaf score; never rewards a dead-end as success."""
        if self.terminal(state):
            return 0.0
        goal_dist = abs(state.x - self.goal[0]) + abs(state.y - self.goal[1])
        key_dist = 0 if state.has_key else abs(state.x - self.key_pos[0]) + abs(state.y - self.key_pos[1])
        remaining = max(0, self.max_energy - state.energy)
        return 2.0 * int(state.has_key) - 0.35 * goal_dist - 0.2 * key_dist - 0.05 * remaining

    def step(self, state: State, action: str, rng: random.Random):
        if action not in self.actions(state):
            return state, -2.0, self.terminal(state), {"invalid_action": True}
        x, y, key, energy = state.x, state.y, state.has_key, state.energy
        reward, info = -0.12, {"action": action}
        if action == "REST":
            energy = min(self.max_energy, energy + 3)
            reward -= 0.28
        else:
            dx, dy = DELTAS[action]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.width and 0 <= ny < self.height) or (nx, ny) in self.walls:
                reward -= 0.75
                info["blocked"] = True
            else:
                x, y = nx, ny
                energy -= 1
                if (x, y) in self.hazards:
                    if rng.random() < self.hazard_probability:
                        energy -= 3
                        reward -= 1.75
                        info["hazard_hit"] = True
                    else:
                        reward -= 0.25
                        info["hazard_avoided"] = True
                if (x, y) == self.key_pos and not key:
                    key = True
                    reward += 1.25
                    info["key_collected"] = True
                if (x, y) == self.goal:
                    if key:
                        reward += 15.0
                        info["goal_reached"] = True
                    else:
                        reward -= 0.8
                        info["key_required"] = True
        steps = state.steps + 1
        next_state = State(x, y, key, energy, steps)
        done = self.terminal(next_state)
        if energy <= 0:
            reward -= 8.0
            info["failure"] = "energy_depleted"
        elif steps >= self.max_steps and not info.get("goal_reached"):
            reward -= 4.0
            info["failure"] = "step_limit"
        return next_state, reward, done, info


def make_world(seed: int, *, width: int = 6, height: int = 6) -> GridWorld:
    """Create reproducible varied maps while preserving a safe path to key and goal."""
    rng = random.Random(seed)
    start, key, goal = (0, 0), (width // 2, height // 2), (width - 1, height - 1)
    reserved = {start, key, goal}
    candidates = [(x, y) for y in range(height) for x in range(width) if (x, y) not in reserved]
    rng.shuffle(candidates)
    wall_count = max(1, width * height // 10)
    walls = set(candidates[:wall_count])
    corridor = ({(x, 0) for x in range(width)}
                | {(width - 1, y) for y in range(height)}
                | {(0, y) for y in range(height)}
                | {(x, key[1]) for x in range(width)})
    walls -= corridor
    candidates = [p for p in candidates if p not in walls and p not in corridor]
    hazard_count = max(2, width * height // 9)
    hazards = set(candidates[:hazard_count])
    return GridWorld(width, height, walls=walls, hazards=hazards, start=start, key=key, goal=goal)
