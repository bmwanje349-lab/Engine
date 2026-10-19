"""PUCT-based, transposition-aware Monte Carlo tree search.

Problems provide stable Markov state keys, transitions and optional heuristics.
A SearchTree can be retained across real moves to reuse the prior search tree.
"""
from __future__ import annotations

from collections import OrderedDict
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
    prior: float = 0.2

    @property
    def mean(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class Node:
    state: Any
    visits: int = 0
    edges: dict[Any, Edge] = field(default_factory=dict)
    children: dict[tuple[Any, Hashable], "Node"] = field(default_factory=dict)
    priors: dict[Any, float] = field(default_factory=dict)


@dataclass
class SearchTree:
    """Bounded per-environment transposition table reusable across decisions."""

    max_nodes: int = 25000
    root: Node | None = None
    nodes: OrderedDict[Hashable, Node] = field(default_factory=OrderedDict)

    def node_for(self, problem: PlanningProblem, state: Any) -> Node:
        key = problem.key(state)
        node = self.nodes.get(key)
        if node is not None:
            self.nodes.move_to_end(key)
            return node
        node = Node(state)
        self.nodes[key] = node
        self._trim(problem)
        return node

    def _trim(self, problem: PlanningProblem) -> None:
        while len(self.nodes) > max(128, self.max_nodes):
            root_key = problem.key(self.root.state) if self.root is not None else None
            evict_key = next((key for key in self.nodes if key != root_key), None)
            if evict_key is None:
                break
            self.nodes.pop(evict_key, None)

    def advance(self, problem: PlanningProblem, action: Any, next_state: Any) -> None:
        """Promote the actually observed child to the root after a real action."""
        next_key = problem.key(next_state)
        if self.root is not None:
            child = self.root.children.get((action, next_key))
            if child is not None:
                self.root = child
                self.root.state = next_state
                return
        self.root = self.node_for(problem, next_state)


def _safe_prior(prior: Callable[[Any, Any], float] | None, state: Any, action: Any) -> float:
    if prior is None:
        return 0.0
    try:
        value = float(prior(state, action))
        return max(-20.0, min(20.0, value)) if math.isfinite(value) else 0.0
    except (TypeError, ValueError, ArithmeticError):
        return 0.0


def _softmax_priors(node: Node, state: Any, actions: Sequence[Any],
                    prior: Callable[[Any, Any], float] | None,
                    prior_weight: float) -> dict[Any, float]:
    logits = [prior_weight * _safe_prior(prior, state, action) for action in actions]
    peak = max(logits) if logits else 0.0
    weights = [math.exp(max(-60.0, min(0.0, value - peak))) for value in logits]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        weights = [1.0] * len(actions)
        total = float(len(actions) or 1)
    priors = {action: weights[i] / total for i, action in enumerate(actions)}
    node.priors = dict(priors)
    for action, probability in priors.items():
        node.edges.setdefault(action, Edge()).prior = probability
    return priors


def _dirichlet_noise(actions: Sequence[Any], rng: random.Random, alpha: float = 0.3) -> dict[Any, float]:
    if not actions:
        return {}
    samples = [rng.gammavariate(alpha, 1.0) for _ in actions]
    total = sum(samples)
    if not math.isfinite(total) or total <= 0:
        return {action: 1.0 / len(actions) for action in actions}
    return {action: samples[i] / total for i, action in enumerate(actions)}


def _pick_puct(node: Node, state: Any, actions: Sequence[Any], rng: random.Random,
               exploration: float, prior: Callable[[Any, Any], float] | None,
               prior_weight: float, prior_override: dict[Any, float] | None = None) -> Any:
    if not node.priors or any(action not in node.priors for action in actions):
        _softmax_priors(node, state, actions, prior, prior_weight)
    priors = prior_override if prior_override is not None else node.priors
    for action in actions:
        node.edges.setdefault(action, Edge()).prior = max(0.0, float(priors.get(action, 0.0)))
    root_scale = math.sqrt(max(1, node.visits))
    best_score, tied = -float("inf"), []
    for action in actions:
        edge = node.edges[action]
        score = edge.mean + exploration * edge.prior * root_scale / (1 + edge.visits)
        if score > best_score + 1e-12:
            best_score, tied = score, [action]
        elif abs(score - best_score) <= 1e-12:
            tied.append(action)
    return rng.choice(tied)


def plan(problem: PlanningProblem, state: Any, *, simulations: int = 400,
         horizon: int = 24, gamma: float = 0.97, exploration: float = 1.5,
         seed: int = 0, top_k: int = 3,
         action_prior: Callable[[Any, Any], float] | None = None,
         prior_weight: float = 1.0, reward_scale: float = 10.0,
         value_evaluator: Callable[[Any], float] | None = None,
         root_noise: float = 0.0, tree: SearchTree | None = None) -> dict[str, Any]:
    """Search alternative futures using PUCT, learned priors and optional leaf values.

    Search effort and horizon are bounded. State/action priors are converted to
    normalized probabilities; root noise supports expert self-play. The optional
    tree table shares repeated states and can be carried between actual moves.
    """
    if (simulations < 1 or horizon < 1 or not 0 < gamma <= 1 or top_k < 1
            or not math.isfinite(reward_scale) or reward_scale <= 0
            or not math.isfinite(exploration) or exploration < 0
            or not math.isfinite(prior_weight) or prior_weight < 0
            or not math.isfinite(root_noise) or not 0 <= root_noise <= 1):
        raise ValueError("search budget, PUCT settings or reward scale are outside valid bounds")
    rng = random.Random(seed)
    tree = tree if tree is not None else SearchTree()
    state_key = problem.key(state)
    if tree.root is None or problem.key(tree.root.state) != state_key:
        tree.root = tree.node_for(problem, state)
    else:
        tree.root.state = state
    root = tree.root
    if problem.terminal(state):
        return {"best_action": None, "alternatives": [], "simulations": simulations,
                "horizon": horizon, "terminal": True, "transposition_nodes": len(tree.nodes)}
    root_actions = list(problem.actions(state))
    if not root_actions:
        return {"best_action": None, "alternatives": [], "simulations": simulations,
                "horizon": horizon, "terminal": True, "transposition_nodes": len(tree.nodes)}

    base_root_priors = _softmax_priors(root, state, root_actions, action_prior, prior_weight)
    root_priors = dict(base_root_priors)
    if root_noise > 0:
        noise = _dirichlet_noise(root_actions, rng)
        root_priors = {
            action: (1.0 - root_noise) * base_root_priors[action] + root_noise * noise[action]
            for action in root_actions
        }

    for _ in range(simulations):
        node, current = root, state
        path: list[tuple[Node, Any, float]] = []
        depth = 0
        leaf_is_new = False
        while depth < horizon and not problem.terminal(current):
            legal = list(problem.actions(current))
            if not legal:
                break
            override = root_priors if node is root else None
            action = _pick_puct(
                node, current, legal, rng, exploration, action_prior, prior_weight, override
            )
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
            leaf_is_new = child is None
            if child is None:
                child = tree.node_for(problem, next_state)
                node.children[child_key] = child
            node, current = child, next_state
            depth += 1
            if leaf_is_new or done:
                break

        # Use the state-value head at expanded/horizon leaves. Without a model,
        # bounded randomized rollouts remain the baseline leaf estimator.
        rollout_return, factor, rollout_depth = 0.0, 1.0, 0
        evaluate_leaf = value_evaluator is not None and not problem.terminal(current)
        if evaluate_leaf:
            leaf = float(value_evaluator(current))
            if not math.isfinite(leaf):
                raise ValueError("neural leaf evaluation must be finite")
            rollout_return = leaf / reward_scale
        else:
            while rollout_depth < horizon - depth and not problem.terminal(current):
                legal = list(problem.actions(current))
                if not legal:
                    break
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
        seen = {problem.key(root.state)}
        for _ in range(min(8, horizon)):
            if current_action not in cursor.edges:
                break
            current_path.append(current_action)
            outcome_nodes = [
                (key, child) for key, child in cursor.children.items() if key[0] == current_action
            ]
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
            "prior": round(edge.prior, 6),
            "visit_share": round(edge.visits / max(1, root.visits), 4),
            "principal_variation": current_path,
        })
    return {
        "best_action": alternatives[0]["first_action"] if alternatives else None,
        "alternatives": alternatives, "simulations": simulations,
        "horizon": horizon, "root_visits": root.visits, "terminal": False,
        "reward_scale": reward_scale, "transposition_nodes": len(tree.nodes),
        "tree_reuse_enabled": tree is not None,
        "selection": "PUCT",
        "warning": "Returns are simulation estimates, not guaranteed outcomes.",
    }
