#!/usr/bin/env python3
"""Train a cross-domain evaluator from search-generated targets and gate it on held-out tasks."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

from strategic_ai.benchmarks import (
    FAMILIES, NimProblem, encode_features, make_problem, nim_action_values,
    nim_hybrid_action_values,
)
from strategic_ai.gridworld import ACTIONS
from strategic_ai.general_neural import GeneralNeuralPolicy
from strategic_ai.search import plan


def _neural_values(problem, state, neural):
    return neural.predict_features(problem.features(state))


def _prior(problem, state, action, neural):
    actions = list(problem.actions(state))
    if not actions:
        return 0.0
    features = problem.features(state)
    if hasattr(neural, "supports_features") and not neural.supports_features(features):
        return 0.0
    values = neural.predict_features(features)
    legal_values = [float(values[ACTIONS.index(a)]) for a in actions]
    center = sum(legal_values) / len(legal_values)
    return max(-1.0, min(1.0, (float(values[ACTIONS.index(action)]) - center) / 1.5))


def _choose_neural(problem, state, neural):
    actions = list(problem.actions(state))
    if not actions:
        return None
    values = _neural_values(problem, state, neural)
    return max(actions, key=lambda a: (values[ACTIONS.index(a)], -ACTIONS.index(a)))


def collect_search_targets(seed: int, target_simulations: int, target_horizon: int,
                           scenarios_per_family: int = 24, states_per_scenario: int = 12,
                           nim_train_max: int = 60):
    """Label reachable states with MCTS or exact adversarial minimax action values."""
    targets: dict[str, dict[str, float]] = {}
    target_states = 0
    target_actions = 0
    # Exact adversarial values are stronger than noisy rollouts for this small,
    # fully observable two-player game. Eval piles are deliberately larger.
    for stones in range(1, nim_train_max + 1):
        problem = NimProblem(stones)
        key = encode_features(problem.features(stones))
        targets[key] = nim_action_values(stones)
        target_states += 1
        target_actions += len(targets[key])

    for family_index, family in enumerate(FAMILIES[:3]):
        for scenario in range(scenarios_per_family):
            problem_seed = seed + family_index * 100000 + scenario * 97
            problem = make_problem(family, problem_seed)
            state = problem.initial_state()
            rng = random.Random(problem_seed + 9001)
            for step_index in range(states_per_scenario):
                if problem.terminal(state):
                    break
                result = plan(
                    problem, state,
                    simulations=target_simulations,
                    horizon=target_horizon,
                    seed=problem_seed + step_index * 13,
                    top_k=5,
                    prior_weight=0.0,
                )
                row = targets.setdefault(encode_features(problem.features(state)), {})
                for alternative in result["alternatives"]:
                    action = alternative["first_action"]
                    score = float(alternative["estimated_return"])
                    if math.isfinite(score):
                        row[action] = score
                target_actions += len(result["alternatives"])
                target_states += 1
                legal = list(problem.actions(state))
                if not legal:
                    break
                # Keep both strong-search states and off-policy branches in the
                # target set so the evaluator does not only see one preferred path.
                action = (rng.choice(legal) if rng.random() < 0.55
                          else result["best_action"])
                if action is None:
                    break
                state, _reward, done, _info = problem.step(state, action, rng)
                if done:
                    break
    clean = {key: row for key, row in targets.items() if row}
    if len(clean) < 30:
        raise RuntimeError(f"search target set is too small: {len(clean)} states")
    return clean, {"unique_target_states": len(clean), "visited_target_states": target_states,
                   "target_action_values": target_actions}


def _episode(problem, family, method, neural, seed, simulations, horizon):
    rng = random.Random(seed)
    state = problem.initial_state()
    total, success, steps, visits = 0.0, False, 0, 0
    limit = 32
    while steps < limit and not problem.terminal(state):
        actions = list(problem.actions(state))
        if not actions:
            break
        if family == "adversarial":
            # The generic MCTS backup is single-agent. Adversarial positions are
            # evaluated with exact negamax or a bounded neural-guided negamax.
            stones = int(state)
            if method == "random":
                action = rng.choice(actions)
            elif method == "neural":
                action = _choose_neural(problem, state, neural)
            elif method == "hybrid":
                values, nodes = nim_hybrid_action_values(stones, neural, depth=5)
                visits += nodes
                action = max(actions, key=lambda a: (values[a], -ACTIONS.index(a)))
            else:
                values = nim_action_values(stones)
                action = max(actions, key=lambda a: (values[a], -ACTIONS.index(a)))
        elif method == "random":
            action = rng.choice(actions)
        elif method == "neural":
            action = _choose_neural(problem, state, neural)
        else:
            prior = None
            if method != "search":
                features = problem.features(state)
                # Do not let out-of-domain neural estimates steer a search.
                if not hasattr(neural, "supports_features") or neural.supports_features(features):
                    prior = lambda s, a: _prior(problem, s, a, neural)
            result = plan(
                problem, state, simulations=simulations, horizon=horizon,
                seed=seed + steps * 17, top_k=5,
                action_prior=prior, prior_weight=(0.0 if prior is None else 0.08),
            )
            visits += int(result.get("root_visits", 0))
            action = result["best_action"]
        if action is None:
            break
        state, reward, done, info = problem.step(state, action, rng)
        total += float(reward)
        steps += 1
        success = success or bool(info.get("success") or info.get("goal_reached"))
        if done:
            break
    if family == "adversarial":
        # A Nim benchmark episode above is one move on a held-out decision state.
        exact_values = nim_action_values(problem.stones)
        chosen = action if steps else None
        success = bool(chosen is not None and abs(exact_values[chosen] - max(exact_values.values())) < 1e-9)
        total = float(exact_values[chosen]) if chosen is not None else -1.0
    return {"return": total, "success": success, "steps": steps, "search_visits": visits}


def _summarize(rows):
    if not rows:
        raise ValueError("cannot summarize an empty benchmark")
    return {
        "episodes": len(rows),
        "success_rate": round(sum(bool(row["success"]) for row in rows) / len(rows), 4),
        "mean_return": round(statistics.fmean(float(row["return"]) for row in rows), 5),
        "mean_steps": round(statistics.fmean(float(row["steps"]) for row in rows), 3),
        "mean_search_visits": round(statistics.fmean(float(row.get("search_visits", 0)) for row in rows), 2),
    }


def evaluate_nim_positions(neural, stones_values=range(61, 81), seed=719):
    """Report decision accuracy and exact minimax regret on unseen pile sizes."""
    methods = {"random": [], "search_only": [], "neural_only": [], "hybrid": []}
    rng = random.Random(seed)
    for stones in stones_values:
        problem = NimProblem(stones)
        legal = list(problem.actions(stones))
        exact = nim_action_values(stones)
        best = max(exact.values())
        predicted = _choose_neural(problem, stones, neural)
        hybrid_values, nodes = nim_hybrid_action_values(stones, neural, depth=5)
        hybrid_action = max(legal, key=lambda a: (hybrid_values[a], -ACTIONS.index(a)))
        search_action = max(legal, key=lambda a: (exact[a], -ACTIONS.index(a)))
        random_action = rng.choice(legal)
        for name, action, count in (
            ("random", random_action, 0),
            ("search_only", search_action, 0),
            ("neural_only", predicted, 0),
            ("hybrid", hybrid_action, nodes),
        ):
            value = exact[action] if action is not None else -1.0
            methods[name].append({
                "return": value,
                "success": bool(action is not None and abs(value - best) < 1e-9),
                "steps": 1,
                "search_visits": count,
                "regret": max(0.0, best - value),
            })
    result = {name: _summarize(rows) for name, rows in methods.items()}
    for name, rows in methods.items():
        result[name]["decision_accuracy"] = result[name]["success_rate"]
        result[name]["mean_exact_regret"] = round(statistics.fmean(r["regret"] for r in rows), 5)
    return result


def evaluate_family(family, neural, *, seed, episodes, simulations, horizon):
    if family == "adversarial":
        return evaluate_nim_positions(neural)
    methods = ("random", "search", "neural", "hybrid")
    results = {}
    family_offset = FAMILIES.index(family) * 1000
    for method in methods:
        rows = []
        for episode in range(episodes):
            problem_seed = seed + family_offset + episode * 31
            problem = make_problem(family, problem_seed)
            rows.append(_episode(problem, family, method, neural,
                                 problem_seed + 50000, simulations, horizon))
        results[method] = _summarize(rows)
    results["search_only"] = results.pop("search")
    results["neural_only"] = results.pop("neural")
    return results


def _family_id_from_key(key):
    fields = key.split(",")
    if len(fields) != 32:
        return None
    domain = [float(value) for value in fields[16:20]]
    if abs(sum(domain) - 1.0) > 1e-5:
        return None
    return max(range(4), key=lambda index: domain[index])


def leave_one_family_out(targets, *, seed, epochs, eval_episodes, simulations, horizon):
    """Cross-validate transfer by training without an entire benchmark family."""
    records = []
    id_to_family = {0: "resource_allocation", 1: "scheduling",
                    2: "route_planning", 3: "adversarial"}
    for heldout_id, family in id_to_family.items():
        fold_targets = {key: row for key, row in targets.items()
                        if _family_id_from_key(key) != heldout_id}
        fold_model, fold_metrics = GeneralNeuralPolicy.distill(
            fold_targets, epochs=epochs, seed=seed + 100 + heldout_id)
        evaluated = evaluate_family(
            family, fold_model, seed=seed + 200000 + heldout_id * 1000,
            episodes=eval_episodes, simulations=simulations, horizon=horizon)
        if family == "adversarial":
            search_return = evaluated["search_only"]["mean_return"]
            hybrid_return = evaluated["hybrid"]["mean_return"]
        else:
            search_return = evaluated["search_only"]["mean_return"]
            hybrid_return = evaluated["hybrid"]["mean_return"]
        records.append({
            "held_out_family": family,
            "training_families": [name for name in FAMILIES if name != family],
            "training_states": fold_metrics["train_states"],
            "validation_states": fold_metrics["validation_states"],
            "neural_validation_mse": fold_metrics["final_validation_mse"],
            "held_out_metrics": evaluated,
            "hybrid_minus_search_return": round(hybrid_return - search_return, 5),
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=410)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--fold-epochs", type=int, default=10)
    parser.add_argument("--episodes-per-family", type=int, default=6)
    parser.add_argument("--target-simulations", type=int, default=24)
    parser.add_argument("--eval-simulations", type=int, default=48)
    parser.add_argument("--target-horizon", type=int, default=10)
    parser.add_argument("--eval-horizon", type=int, default=12)
    parser.add_argument("--fold-eval-episodes", type=int, default=4)
    parser.add_argument("--target-scenarios", type=int, default=24)
    parser.add_argument("--target-states", type=int, default=12)
    parser.add_argument("--train-nim-max", type=int, default=60)
    parser.add_argument("--out", default="artifacts/strategic_general_neural.json")
    parser.add_argument("--metrics", default="artifacts/general_metrics.json")
    args = parser.parse_args()
    if not 4 <= args.episodes_per_family <= 100 or not 8 <= args.target_simulations <= 5000:
        raise SystemExit("scenario count or target search budget outside configured bounds")
    if not 8 <= args.eval_simulations <= 5000 or not 4 <= args.target_horizon <= 100:
        raise SystemExit("evaluation or target horizon outside configured bounds")
    if (not 4 <= args.eval_horizon <= 100 or not 1 <= args.epochs <= 300 or not 1 <= args.fold_epochs <= 100
            or not 4 <= args.target_scenarios <= 100 or not 4 <= args.target_states <= 60
            or not 30 <= args.train_nim_max <= 100):
        raise SystemExit("training epochs or search horizon outside configured bounds")

    started = time.time()
    targets, target_summary = collect_search_targets(
        args.seed, args.target_simulations, args.target_horizon,
        scenarios_per_family=args.target_scenarios, states_per_scenario=args.target_states,
        nim_train_max=args.train_nim_max)
    neural, neural_metrics = GeneralNeuralPolicy.distill(
        targets, epochs=args.epochs, seed=args.seed + 7)
    neural_path = Path(args.out)
    neural.save(neural_path)

    heldout = {}
    for family in FAMILIES:
        heldout[family] = evaluate_family(
            family, neural, seed=args.seed + 300000,
            episodes=args.episodes_per_family,
            simulations=args.eval_simulations, horizon=args.eval_horizon)

    # Entire-family holdouts guard against reporting only fresh seeds from one
    # familiar environment. The hybrid must remain close to search even when
    # its evaluator has never seen that task family.
    folds = leave_one_family_out(
        targets, seed=args.seed, epochs=args.fold_epochs,
        eval_episodes=args.fold_eval_episodes,
        simulations=max(8, min(args.eval_simulations, 24)), horizon=min(args.eval_horizon, 10))

    deltas = {}
    for family, scores in heldout.items():
        deltas[family] = round(
            scores["hybrid"]["mean_return"] - scores["search_only"]["mean_return"], 5)

    family_holdout_min_delta = min(row["hybrid_minus_search_return"] for row in folds)
    aggregate_min_delta = min(deltas.values())
    neural_vs_random = {}
    for family, scores in heldout.items():
        if "random" in scores:
            neural_vs_random[family] = round(
                scores["neural_only"]["mean_return"] - scores["random"]["mean_return"], 5)
        else:
            neural_vs_random[family] = round(
                scores["neural_only"]["mean_return"] - scores["search_only"]["mean_return"], 5)
    neural_baseline_pass = all(value >= 0.0 for value in neural_vs_random.values())
    acceptance = {
        "search_target_validation_mse_improved": bool(neural_metrics["validation_mse_improved"]),
        "neural_only_vs_baseline_return_deltas": neural_vs_random,
        "neural_only_beats_or_matches_baseline_on_every_family": neural_baseline_pass,
        "held_out_hybrid_vs_search_deltas": deltas,
        "leave_one_family_out_hybrid_min_delta": family_holdout_min_delta,
        "all_metrics_finite": all(
            math.isfinite(float(scores[method]["mean_return"]))
            and 0.0 <= float(scores[method]["success_rate"]) <= 1.0
            for scores in heldout.values()
            for method in scores
        ),
        "promote_candidate": bool(
            neural_metrics["validation_mse_improved"]
            and neural_baseline_pass
            and aggregate_min_delta >= -2.0
            and family_holdout_min_delta >= -2.0
        ),
    }
    metrics = {
        "model": "StrategicReasoner-GeneralSearchDistilled-v3",
        "training": {
            "families": list(FAMILIES),
            "target_generation": target_summary,
            "epochs": args.epochs,
            "neural_distillation": neural_metrics,
        },
        "held_out_scenarios": heldout,
        "leave_one_family_out": folds,
        "acceptance": acceptance,
        "checkpoint": {"path": str(neural_path), "bytes": neural_path.stat().st_size},
        "runtime_seconds": round(time.time() - started, 2),
        "validation_scope": (
            "synthetic resource allocation, job scheduling, weighted route planning, "
            "and adversarial take-away positions; seeded held-out scenarios and "
            "leave-one-family-out transfer. Not real-world or general-intelligence evidence."
        ),
    }
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
                            encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))
    if not acceptance["all_metrics_finite"]:
        raise SystemExit("held-out evaluation contained invalid metrics; candidate rejected")
    if not acceptance["promote_candidate"]:
        raise SystemExit("general evaluator failed held-out regression gates; candidate rejected")


if __name__ == "__main__":
    main()
