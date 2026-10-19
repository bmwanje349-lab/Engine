"""Model-based Monte Carlo tree search for finite-horizon decision problems.

The planner is deliberately domain-agnostic. A problem supplies legal actions,
transitions, terminal checks, stable state keys and an optional leaf evaluator.
All transition rewards are discounted during backup; stochastic transitions are
represented as separate child outcomes beneath the same parent action.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Callable, Hashable, Protocol, Sequence


class PlanningProblem(Protocol):
    def actions(self, state: Any) -> Sequence[Any]: ...
    def step(self, state: Any, action: Any, rng: random.Random) -> tuple[Any, float, bool, dict]: ...
    def terminal(self, state: Any) -> bool: ...
    def key(self, state: Any) -> Hashable: ...
    def evaluate(self, state: Any) -> float: ...


@dataclass
class Edge:
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class Node:
    state: Any
    visits: int = 0
    edges: dict[Any, Edge] = field(default_factory=dict)
    children: dict[tuple[Any, Hashable], "Node"] = field(default_factory=dict)


def _safe_prior(prior: Callable[[Any, Any], float] | None, state: Any, action: Any) -> float:
    if prior is None:
        return 0.0
    try:
        value = float(prior(state, action))
        return max(-5.0, min(5.0, value)) if math.isfinite(value) else 0.0
    except (TypeError, ValueError, ArithmeticError):
        return 0.0


def _pick_uct(node: Node, actions: Sequence[Any], rng: random.Random,
              exploration: float, prior: Callable[[Any, Any], float] | None,
              prior_weight: float) -> Any:
    for action in actions:
        node.edges.setdefault(action, Edge())
    unexplored = [a for a in actions if node.edges[a].visits == 0]
    if unexplored:
        # Expand the strongest learned prior first, but still give every legal
        # action an initial simulation before ordinary UCT exploitation begins.
        if prior is not None:
            scored = [(action, _safe_prior(prior, node.state, action)) for action in unexplored]
            best = max(score for _, score in scored)
            return rng.choice([a for a, score in scored if abs(score - best) <= 1e-12])
        return rng.choice(unexplored)
    log_n = math.log(max(2, node.visits + 1))
    best_score, tied = -float("inf"), []
    for action in actions:
        edge = node.edges[action]
        score = edge.mean + exploration * math.sqrt(log_n / edge.visits)
        score += prior_weight * _safe_prior(prior, node.state, action) / (1 + edge.visits)
        if score > best_score + 1e-12:
            best_score, tied = score, [action]
        elif abs(score - best_score) <= 1e-12:
            tied.append(action)
    return rng.choice(tied)


def plan(problem: PlanningProblem, state: Any, *, simulations: int = 400,
         horizon: int = 24, gamma: float = 0.97, exploration: float = 1.35,
         seed: int = 0, top_k: int = 3,
         action_prior: Callable[[Any, Any], float] | None = None,
         prior_weight: float = 0.15, reward_scale: float = 10.0) -> dict[str, Any]:
    """Search alternative futures and return ranked first actions plus paths.

    simulations bounds search effort and horizon bounds every simulated
    future. reward_scale keeps UCT's exploration bonus meaningful when reward
    magnitudes vary. Reported estimated returns are rescaled to reward units.
    Values are estimates from the supplied transition model, not guarantees.
    Actions and state keys must be hashable.
    """
    if (simulations < 1 or horizon < 1 or not 0 < gamma <= 1 or top_k < 1
            or not math.isfinite(reward_scale) or reward_scale <= 0):
        raise ValueError("simulations, horizon, top_k, gamma and reward_scale are outside valid bounds")
    rng = random.Random(seed)
    root = Node(state)
    if problem.terminal(state):
        return {"best_action": None, "alternatives": [], "simulations": simulations,
                "horizon": horizon, "terminal": True}
    root_actions = list(problem.actions(state))
    if not root_actions:
        return {"best_action": None, "alternatives": [], "simulations": simulations,
                "horizon": horizon, "terminal": True}

    for _ in range(simulations):
        node, current = root, state
        path: list[tuple[Node, Any, float]] = []
        depth = 0
        # Selection/expansion. Separate children preserve sampled outcomes from
        # stochastic transitions, even when the parent action is the same.
        while depth < horizon and not problem.terminal(current):
            legal = list(problem.actions(current))
            if not legal:
                break
            action = _pick_uct(node, legal, rng, exploration, action_prior, prior_weight)
            next_state, reward, done, _info = problem.step(current, action, rng)
            try:
                reward = float(reward)
            except (TypeError, ValueError) as exc:
                raise ValueError("transition reward must be numeric") from exc
            if not math.isfinite(reward):
                raise ValueError("transition reward must be finite")
            reward /= reward_scale
            path.append((node, action, reward))
            child_key = (action, problem.key(next_state))
            child = node.children.get(child_key)
            is_new = child is None
            if is_new:
                child = Node(next_state)
                node.children[child_key] = child
            node, current = child, next_state
            depth += 1
            if is_new or done:
                break

        # Randomized / prior-guided rollouts compare futures instead of fixing a
        # single route. The rollout is always bounded by the total horizon.
        rollout_return, factor, rollout_depth = 0.0, 1.0, 0
        while rollout_depth < horizon - depth and not problem.terminal(current):
            legal = list(problem.actions(current))
            if not legal:
                break
            # Keep a learned prior advisory during rollouts; a bad transfer prior must not dominate search.
            if action_prior is not None and rng.random() < 0.25:
                action = max(legal, key=lambda a: _safe_prior(action_prior, current, a))
            else:
                action = rng.choice(legal)
            current, reward, done, _info = problem.step(current, action, rng)
            try:
                reward = float(reward)
            except (TypeError, ValueError) as exc:
                raise ValueError("transition reward must be numeric") from exc
            if not math.isfinite(reward):
                raise ValueError("transition reward must be finite")
            rollout_return += factor * (reward / reward_scale)
            factor *= gamma
            rollout_depth += 1
            if done:
                break
        if not problem.terminal(current) and depth + rollout_depth >= horizon:
            leaf = float(problem.evaluate(current))
            if not math.isfinite(leaf):
                raise ValueError("leaf evaluation must be finite")
            rollout_return += factor * (leaf / reward_scale)

        # Back-propagate discounted future return to every decision on the path.
        future = rollout_return
        for parent, action, reward in reversed(path):
            future = reward + gamma * future
            edge = parent.edges[action]
            edge.visits += 1
            edge.value_sum += future
            parent.visits += 1

    ranked = sorted(root.edges.items(), key=lambda item: (item[1].mean, item[1].visits), reverse=True)
    alternatives = []
    for action, edge in ranked[:top_k]:
        cursor = root
        current_path = []
        current_action = action
        seen = set()
        for _ in range(min(8, horizon)):
            if current_action not in cursor.edges:
                break
            current_path.append(current_action)
            outcome_nodes = [(key, child) for key, child in cursor.children.items() if key[0] == current_action]
            if not outcome_nodes:
                break
            _key, child = max(outcome_nodes, key=lambda item: item[1].visits)
            state_key = problem.key(child.state)
            if state_key in seen or problem.terminal(child.state):
                break
            seen.add(state_key)
            legal_next = list(problem.actions(child.state))
            if not legal_next:
                break
            visited_next = [(a, child.edges.get(a, Edge())) for a in legal_next]
            current_action = max(visited_next, key=lambda item: (item[1].visits, item[1].mean))[0]
            cursor = child
        alternatives.append({
            "first_action": action,
            "estimated_return": round(edge.mean * reward_scale, 5),
            "visits": edge.visits,
            "visit_share": round(edge.visits / max(1, root.visits), 4),
            "principal_variation": current_path,
        })
    return {"best_action": alternatives[0]["first_action"] if alternatives else None,
            "alternatives": alternatives, "simulations": simulations,
            "horizon": horizon, "root_visits": root.visits, "terminal": False,
            "reward_scale": reward_scale,
            "warning": "Returns are simulation estimates, not guaranteed outcomes."}
