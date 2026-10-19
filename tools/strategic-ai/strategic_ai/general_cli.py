"""Pure-Python multi-domain decision CLI suitable for Android Termux."""
from __future__ import annotations

import argparse
import json
import sys

from .action_encoding import encode_action_features
from .benchmarks import FAMILIES, make_problem, nim_action_values, nim_hybrid_action_values
from .gridworld import ACTIONS
from .general_neural import GeneralNeuralPolicy, normalized_action_prior
from .search import SearchTree, plan


def _action_values(problem, state, neural):
    features = problem.features(state)
    return {
        action: neural.predict_action_value(
            features, encode_action_features(features, action), action_name=action
        )
        for action in problem.actions(state)
    }


def _leaf_value(problem, state, neural):
    features = problem.features(state)
    if hasattr(neural, "supports_features") and not neural.supports_features(features):
        return float(problem.evaluate(state))
    if hasattr(neural, "predict_value_features"):
        return float(neural.predict_value_features(features))
    return max(_action_values(problem, state, neural).values(), default=0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, default="route_planning")
    parser.add_argument("--mode", choices=("hybrid", "search", "neural"), default="hybrid")
    parser.add_argument("--model", default="artifacts/strategic_general_neural.json")
    parser.add_argument("--seed", type=int, default=410)
    parser.add_argument("--simulations", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=16)
    args = parser.parse_args()
    if not 8 <= args.simulations <= 5000 or not 4 <= args.horizon <= 100:
        parser.error("simulations must be 8..5000 and horizon must be 4..100")
    try:
        neural = GeneralNeuralPolicy.load(args.model)
        problem = make_problem(args.family, args.seed)
        state = problem.initial_state()
        legal = list(problem.actions(state))
        if not legal:
            decision = {"best_action": None, "alternatives": [], "terminal": True}
        elif args.family == "adversarial":
            if args.mode == "search":
                values = nim_action_values(int(state))
                nodes = None
            elif args.mode == "hybrid":
                values, nodes = nim_hybrid_action_values(
                    int(state), neural, depth=min(10, args.horizon)
                )
            else:
                values = _action_values(problem, state, neural)
                nodes = 0
            ranked = sorted(
                ({"first_action": action, "estimated_return": round(float(values[action]), 5)}
                 for action in legal),
                key=lambda row: row["estimated_return"], reverse=True,
            )
            decision = {
                "best_action": ranked[0]["first_action"], "alternatives": ranked,
                "terminal": False, "search_nodes": nodes,
                "warning": "Adversarial values are synthetic-game estimates; exact minimax is available in search mode.",
            }
        elif args.mode == "neural":
            values = _action_values(problem, state, neural)
            ranked = sorted(
                ({"first_action": action, "estimated_return": round(float(values[action]), 5)}
                 for action in legal),
                key=lambda row: row["estimated_return"], reverse=True,
            )
            decision = {"best_action": ranked[0]["first_action"], "alternatives": ranked,
                        "terminal": False, "selection": "action-conditioned Q(s,a)"}
        else:
            prior = None
            value_evaluator = None
            if args.mode == "hybrid":
                def prior(s, action):
                    features = problem.features(s)
                    if hasattr(neural, "supports_features") and not neural.supports_features(features):
                        return 0.0
                    return normalized_action_prior(_action_values(problem, s, neural), action)
                value_evaluator = lambda s: _leaf_value(problem, s, neural)
            tree = SearchTree()
            result = plan(
                problem, state, simulations=args.simulations, horizon=args.horizon,
                seed=args.seed, top_k=5, action_prior=prior,
                prior_weight=(0.75 if prior is not None else 0.0),
                value_evaluator=value_evaluator, tree=tree,
            )
            decision = result
        value_estimate = _leaf_value(problem, state, neural)
        print(json.dumps({
            "engine": "Strategic Reasoning AI",
            "family": args.family,
            "mode": args.mode,
            "state": state,
            "features": problem.features(state),
            "legal_actions": legal,
            "state_value_estimate": round(float(value_estimate), 5),
            "decision": decision,
            "validation_notice": "Experimental synthetic planner; evaluate before relying on decisions.",
        }, indent=2, sort_keys=True))
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
