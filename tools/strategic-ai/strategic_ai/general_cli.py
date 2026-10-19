"""Pure-Python multi-domain decision CLI suitable for Android Termux."""
from __future__ import annotations

import argparse
import json
import sys

from .benchmarks import FAMILIES, NimProblem, make_problem, nim_action_values, nim_hybrid_action_values
from .gridworld import ACTIONS
from .neural import NeuralPolicy
from .search import plan


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
        neural = NeuralPolicy.load(args.model)
        problem = make_problem(args.family, args.seed)
        state = problem.initial_state()
        legal = list(problem.actions(state))
        if not legal:
            decision = {"best_action": None, "alternatives": [], "terminal": True}
        elif args.family == "adversarial":
            stones = int(state)
            if args.mode == "search":
                values = nim_action_values(stones)
                nodes = None
            elif args.mode == "hybrid":
                values, nodes = nim_hybrid_action_values(stones, neural, depth=min(10, args.horizon))
            else:
                predicted = neural.predict_features(problem.features(state))
                values = {a: float(predicted[ACTIONS.index(a)]) for a in legal}
                nodes = 0
            ranked = sorted(
                ({"first_action": a, "estimated_return": round(float(values[a]), 5)}
                 for a in legal),
                key=lambda row: row["estimated_return"], reverse=True)
            decision = {"best_action": ranked[0]["first_action"], "alternatives": ranked,
                        "terminal": False, "search_nodes": nodes,
                        "warning": "Adversarial values are synthetic-game estimates; exact minimax is available in search mode."}
        elif args.mode == "neural":
            predicted = neural.predict_features(problem.features(state))
            ranked = sorted(
                ({"first_action": a, "estimated_return": round(float(predicted[ACTIONS.index(a)]), 5)}
                 for a in legal),
                key=lambda row: row["estimated_return"], reverse=True)
            decision = {"best_action": ranked[0]["first_action"], "alternatives": ranked,
                        "terminal": False}
        else:
            prior = None
            if args.mode == "hybrid":
                def prior(s, a):
                    values = neural.predict_features(problem.features(s))
                    legal_values = [values[ACTIONS.index(item)] for item in problem.actions(s)]
                    center = sum(legal_values) / max(1, len(legal_values))
                    return max(-1.0, min(1.0, (values[ACTIONS.index(a)] - center) / 5.0))
            result = plan(
                problem, state, simulations=args.simulations, horizon=args.horizon,
                seed=args.seed, top_k=5, action_prior=prior,
                prior_weight=(0.0 if prior is None else 0.035))
            decision = result
        print(json.dumps({
            "engine": "Strategic Reasoning AI",
            "family": args.family,
            "mode": args.mode,
            "state": state,
            "features": problem.features(state),
            "legal_actions": legal,
            "decision": decision,
            "validation_notice": "Experimental synthetic planner; evaluate before relying on decisions.",
        }, indent=2, sort_keys=True))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
