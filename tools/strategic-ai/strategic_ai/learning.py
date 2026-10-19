"""Tabular Q-learning and privacy-safe checkpoint I/O for the reference domain.

The learned table is a compact warm-start prior for search. Search remains in
control and can overturn a stale prior when simulated future return disagrees.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import random
from collections import defaultdict
from typing import Any

from .gridworld import ACTIONS, GridWorld, State


def state_features(world: GridWorld, state: State) -> tuple[int, ...]:
    # Relative coordinates transfer better across maps than raw absolute goals.
    dx_goal = world.goal[0] - state.x
    dy_goal = world.goal[1] - state.y
    dx_key = 0 if state.has_key else world.key_pos[0] - state.x
    dy_key = 0 if state.has_key else world.key_pos[1] - state.y
    energy_band = min(5, state.energy // max(1, world.max_energy // 6))
    steps_band = min(5, state.steps // max(1, world.max_steps // 6))
    local = []
    deltas = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}
    for action in ("UP", "RIGHT", "DOWN", "LEFT"):
        dx, dy = deltas[action]
        p = (state.x + dx, state.y + dy)
        blocked = not (0 <= p[0] < world.width and 0 <= p[1] < world.height) or p in world.walls
        local.append(int(blocked))
    return (dx_goal, dy_goal, dx_key, dy_key, int(state.has_key), energy_band, steps_band, *local)


def encode(features: tuple[int, ...]) -> str:
    return ",".join(str(v) for v in features)


class QLearner:
    def __init__(self, *, alpha: float = 0.18, gamma: float = 0.96, epsilon: float = 0.22,
                 min_epsilon: float = 0.03, epsilon_decay: float = 0.995, seed: int = 17):
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.min_epsilon, self.epsilon_decay = epsilon, min_epsilon, epsilon_decay
        self.rng = random.Random(seed)
        self.q: dict[str, dict[str, float]] = defaultdict(dict)
        self.updates = 0

    def values(self, world: GridWorld, state: State) -> dict[str, float]:
        row = self.q.get(encode(state_features(world, state)), {})
        return {a: float(row.get(a, 0.0)) for a in world.actions(state)}

    def prior(self, world: GridWorld, state: State, action: str) -> float:
        vals = self.values(world, state)
        value = vals.get(action, 0.0)
        if not vals:
            return 0.0
        avg = sum(vals.values()) / len(vals)
        return max(-2.0, min(2.0, value - avg))

    def choose(self, world: GridWorld, state: State, *, explore: bool) -> str:
        legal = list(world.actions(state))
        if not legal:
            raise ValueError("no actions available in terminal state")
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(legal)
        values = self.values(world, state)
        best = max(values.values())
        ties = sorted(a for a in legal if abs(values[a] - best) < 1e-9)
        return self.rng.choice(ties) if explore else ties[0]

    def update(self, world: GridWorld, state: State, action: str, reward: float,
               next_state: State, done: bool) -> None:
        key = encode(state_features(world, state))
        row = self.q.setdefault(key, {})
        old = float(row.get(action, 0.0))
        next_values = self.values(world, next_state) if not done else {}
        target = float(reward) + (self.gamma * max(next_values.values()) if next_values else 0.0)
        row[action] = old + self.alpha * (target - old)
        self.updates += 1

    def train_episode(self, world: GridWorld, *, max_steps: int | None = None) -> dict[str, Any]:
        state = world.initial_state()
        total, success, trace = 0.0, False, []
        limit = max_steps or world.max_steps
        for _ in range(limit):
            if world.terminal(state):
                break
            action = self.choose(world, state, explore=True)
            next_state, reward, done, info = world.step(state, action, self.rng)
            self.update(world, state, action, reward, next_state, done)
            total += reward
            trace.append({"state": encode(state_features(world, state)), "action": action,
                          "reward": round(float(reward), 5), "done": bool(done)})
            state = next_state
            success = success or bool(info.get("goal_reached"))
            if done:
                break
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return {"return": round(total, 5), "success": success, "steps": len(trace), "trace": trace}

    def evaluate_episode(self, world: GridWorld, *, seed: int = 0, max_steps: int | None = None) -> dict[str, Any]:
        rng = random.Random(seed)
        state, total, success, steps = world.initial_state(), 0.0, False, 0
        for _ in range(max_steps or world.max_steps):
            if world.terminal(state):
                break
            action = self.choose(world, state, explore=False)
            state, reward, done, info = world.step(state, action, rng)
            total += reward
            steps += 1
            success = success or bool(info.get("goal_reached"))
            if done:
                break
        return {"return": round(total, 5), "success": success, "steps": steps}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": "strategic-ai-qtable-v1", "alpha": self.alpha, "gamma": self.gamma,
                   "epsilon": self.epsilon, "updates": self.updates,
                   "states": {k: {a: float(v) for a, v in sorted(row.items())}
                              for k, row in sorted(self.q.items())}}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path, *, seed: int = 17) -> "QLearner":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "strategic-ai-qtable-v1" or not isinstance(payload.get("states"), dict):
            raise ValueError("unsupported or malformed Strategic AI checkpoint")
        model = cls(alpha=float(payload.get("alpha", 0.18)), gamma=float(payload.get("gamma", 0.96)),
                    epsilon=float(payload.get("epsilon", 0.03)), seed=seed)
        for key, row in payload["states"].items():
            if not isinstance(key, str) or not isinstance(row, dict):
                raise ValueError("malformed Q-table row")
            clean = {}
            for action, value in row.items():
                if action not in ACTIONS or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("invalid action/value in checkpoint")
                clean[action] = float(value)
            model.q[key] = clean
        model.updates = int(payload.get("updates", 0))
        return model
