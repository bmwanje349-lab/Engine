#!/usr/bin/env python3
"""Expert-iteration training and confidence-bound gates for cross-domain planning."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

from strategic_ai.action_encoding import encode_action_features
from strategic_ai.benchmarks import (
    FAMILIES, NimProblem, encode_features, make_problem, nim_action_values,
    nim_hybrid_action_values,
)
from strategic_ai.general_neural import GeneralNeuralPolicy, _split_target_key, normalized_action_prior
from strategic_ai.search import SearchTree, plan


METHODS = ("random", "search_only", "neural_only", "hybrid")


def _neural_values(problem, state, neural):
    features = problem.features(state)
    result = {}
    for action in problem.actions(state):
        action_features = encode_action_features(features, action)
        result[action] = neural.predict_action_value(
            features, action_features, action_name=action
        )
    return result


def _prior(problem, state, action, neural):
    features = problem.features(state)
    if hasattr(neural, "supports_features") and not neural.supports_features(features):
        return 0.0
    # Normalize Q(s,a) against this state's legal alternatives before using
    # it as a prior; raw rewards vary sharply across domains.
    return normalized_action_prior(_neural_values(problem, state, neural), action)


def _leaf_value(problem, state, neural):
    features = problem.features(state)
    if hasattr(neural, "supports_features") and not neural.supports_features(features):
        return float(problem.evaluate(state))
    if hasattr(neural, "predict_value_features"):
        return float(neural.predict_value_features(features))
    return max(_neural_values(problem, state, neural).values(), default=0.0)


def _choose_neural(problem, state, neural):
    actions = list(problem.actions(state))
    if not actions:
        return None
    values = _neural_values(problem, state, neural)
    return max(actions, key=lambda action: (values[action], -actions.index(action)))


def _target_key(scenario_id, features):
    return f"{scenario_id}|{encode_features(features)}"


def collect_search_targets(seed: int, target_simulations: int, target_horizon: int,
                           scenarios_per_family: int = 24, states_per_scenario: int = 12,
                           nim_train_max: int = 60, guide=None, iteration: int = 0):
    """Generate search expert targets, optionally using the current net in PUCT and leaf values."""
    targets: dict[str, dict[str, float]] = {}
    visited_states = 0
    target_actions = 0
    # Exact minimax labels are stronger than rollouts for this finite adversarial task.
    for stones in range(1, nim_train_max + 1):
        problem = NimProblem(stones)
        key = _target_key(f"nim:{stones}", problem.features(stones))
        targets[key] = nim_action_values(stones)
        visited_states += 1
        target_actions += len(targets[key])

    for family_index, family in enumerate(FAMILIES[:3]):
        for scenario in range(scenarios_per_family):
            problem_seed = seed + family_index * 100000 + scenario * 97 + iteration * 1000003
            scenario_id = f"iter{iteration}:{family}:difficulty{1 if scenario % 4 == 0 else 0}:{problem_seed}"
            # Mix base scenarios with medium-difficulty tasks during training;
            # the promotion-time stress suite remains strictly harder.
            difficulty = 1 if scenario % 4 == 0 else 0
            problem = make_problem(family, problem_seed, difficulty=difficulty)
            state = problem.initial_state()
            rng = random.Random(problem_seed + 9001)
            tree = SearchTree()
            for step_index in range(states_per_scenario):
                if problem.terminal(state):
                    break
                prior = None
                value_evaluator = None
                if guide is not None:
                    features = problem.features(state)
                    if not hasattr(guide, "supports_features") or guide.supports_features(features):
                        prior = lambda s, a, p=problem: _prior(p, s, a, guide)
                        value_evaluator = lambda s, p=problem: _leaf_value(p, s, guide)
                result = plan(
                    problem, state, simulations=target_simulations, horizon=target_horizon,
                    seed=problem_seed + step_index * 13, top_k=5,
                    action_prior=prior, prior_weight=(0.75 if prior is not None else 0.0),
                    value_evaluator=value_evaluator, root_noise=(0.20 if guide is not None else 0.0),
                    tree=tree,
                )
                key = _target_key(scenario_id, problem.features(state))
                row = targets.setdefault(key, {})
                for alternative in result["alternatives"]:
                    action = alternative["first_action"]
                    score = float(alternative["estimated_return"])
                    if math.isfinite(score):
                        row[action] = score
                target_actions += len(result["alternatives"])
                visited_states += 1
                legal = list(problem.actions(state))
                if not legal:
                    break
                best_action = result["best_action"]
                # Expert iteration follows search most of the time but keeps
                # off-policy paths for broader state coverage.
                action = (best_action if best_action is not None and rng.random() < 0.75
                          else rng.choice(legal))
                next_state, _reward, done, _info = problem.step(state, action, rng)
                tree.advance(problem, action, next_state)
                state = next_state
                if done:
                    break
    clean = {key: row for key, row in targets.items() if row}
    if len(clean) < 30:
        raise RuntimeError(f"search target set is too small: {len(clean)} states")
    scenario_groups = {key.split("|", 1)[0] if "|" in key else key for key in clean}
    return clean, {
        "unique_target_states": len(clean),
        "visited_target_states": visited_states,
        "target_action_values": target_actions,
        "scenario_groups": len(scenario_groups),
        "scenario_groups_by_family_target": scenarios_per_family,
        "iteration": iteration,
        "guided_by_model": guide is not None,
    }


def _episode(problem, family, method, neural, seed, simulations, horizon):
    rng = random.Random(seed)
    state = problem.initial_state()
    total, success, steps, visits = 0.0, False, 0, 0
    limit = 32
    tree = SearchTree() if method in ("search", "hybrid") and family != "adversarial" else None
    chosen_action = None
    while steps < limit and not problem.terminal(state):
        actions = list(problem.actions(state))
        if not actions:
            break
        if family == "adversarial":
            stones = int(state)
            if method == "random":
                action = rng.choice(actions)
            elif method == "neural":
                action = _choose_neural(problem, state, neural)
            elif method == "hybrid":
                values, nodes = nim_hybrid_action_values(stones, neural, depth=5)
                visits += nodes
                action = max(actions, key=lambda a: (values[a], -actions.index(a)))
            else:
                values = nim_action_values(stones)
                action = max(actions, key=lambda a: (values[a], -actions.index(a)))
        elif method == "random":
            action = rng.choice(actions)
        elif method == "neural":
            action = _choose_neural(problem, state, neural)
        else:
            prior = None
            value_evaluator = None
            if method == "hybrid":
                features = problem.features(state)
                if not hasattr(neural, "supports_features") or neural.supports_features(features):
                    prior = lambda s, a: _prior(problem, s, a, neural)
                    value_evaluator = lambda s: _leaf_value(problem, s, neural)
            result = plan(
                problem, state, simulations=simulations, horizon=horizon,
                seed=seed + steps * 17, top_k=5, action_prior=prior,
                prior_weight=(0.75 if prior is not None else 0.0),
                value_evaluator=value_evaluator, tree=tree,
            )
            visits += int(result.get("root_visits", 0))
            action = result["best_action"]
        if action is None:
            break
        chosen_action = action
        next_state, reward, done, info = problem.step(state, action, rng)
        if tree is not None:
            tree.advance(problem, action, next_state)
        state = next_state
        total += float(reward)
        steps += 1
        success = success or bool(info.get("success") or info.get("goal_reached"))
        if done:
            break
    if family == "adversarial":
        # Score the chosen first move against exact minimax value at this position.
        exact_values = nim_action_values(problem.stones)
        success = bool(chosen_action is not None
                       and abs(exact_values[chosen_action] - max(exact_values.values())) < 1e-9)
        total = float(exact_values[chosen_action]) if chosen_action is not None else -1.0
    return {"return": total, "success": success, "steps": steps, "search_visits": visits}


def _ci95(values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "n": 0}
    mean = statistics.fmean(values)
    if len(values) < 2:
        margin = 0.0
    else:
        # Evaluation uses >=40 independent held-out positions/episodes; 1.96
        # is a conservative large-sample 95% confidence multiplier.
        margin = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return {
        "mean": round(mean, 5),
        "lower": round(mean - margin, 5),
        "upper": round(mean + margin, 5),
        "n": len(values),
    }


def _paired_summary(left_rows, right_rows, metric="return"):
    count = min(len(left_rows), len(right_rows))
    differences = [
        float(left_rows[i][metric]) - float(right_rows[i][metric])
        for i in range(count)
    ]
    ci = _ci95(differences)
    return {
        "episodes": count,
        "mean_delta": ci["mean"],
        "ci95_lower": ci["lower"],
        "ci95_upper": ci["upper"],
    }


def _summarize(rows):
    if not rows:
        raise ValueError("cannot summarize an empty benchmark")
    return {
        "episodes": len(rows),
        "success_rate": round(sum(bool(row["success"]) for row in rows) / len(rows), 4),
        "mean_return": round(statistics.fmean(float(row["return"]) for row in rows), 5),
        "mean_return_ci95": {
            "lower": _ci95([row["return"] for row in rows])["lower"],
            "upper": _ci95([row["return"] for row in rows])["upper"],
        },
        "mean_steps": round(statistics.fmean(float(row["steps"]) for row in rows), 3),
        "mean_search_visits": round(statistics.fmean(float(row.get("search_visits", 0)) for row in rows), 2),
    }


def evaluate_nim_positions(neural, stones_values=range(61, 101), seed=719, return_rows=False):
    """Report exact minimax accuracy/regret on 40 unseen pile sizes with paired comparisons."""
    runs = {"random": [], "search_only": [], "neural_only": [], "hybrid": []}
    rng = random.Random(seed)
    for stones in stones_values:
        problem = NimProblem(stones)
        legal = list(problem.actions(stones))
        exact = nim_action_values(stones)
        best = max(exact.values())
        predicted = _choose_neural(problem, stones, neural)
        hybrid_values, nodes = nim_hybrid_action_values(stones, neural, depth=5)
        hybrid_action = max(legal, key=lambda a: (hybrid_values[a], -legal.index(a)))
        search_action = max(legal, key=lambda a: (exact[a], -legal.index(a)))
        random_action = rng.choice(legal)
        for name, action, count in (
            ("random", random_action, 0),
            ("search_only", search_action, 0),
            ("neural_only", predicted, 0),
            ("hybrid", hybrid_action, nodes),
        ):
            value = exact[action] if action is not None else -1.0
            runs[name].append({
                "return": value,
                "success": bool(action is not None and abs(value - best) < 1e-9),
                "steps": 1,
                "search_visits": count,
                "regret": max(0.0, best - value),
            })
    result = {name: _summarize(rows) for name, rows in runs.items()}
    for name, rows in runs.items():
        result[name]["decision_accuracy"] = result[name]["success_rate"]
        result[name]["mean_exact_regret"] = round(statistics.fmean(r["regret"] for r in rows), 5)
    result["paired_comparisons"] = {
        "hybrid_minus_search": _paired_summary(runs["hybrid"], runs["search_only"]),
        "neural_minus_random": _paired_summary(runs["neural_only"], runs["random"]),
    }
    if return_rows:
        return result, runs
    return result


def evaluate_family(family, neural, *, seed, episodes, simulations, horizon,
                    return_rows=False, difficulty=0):
    if family == "adversarial":
        return evaluate_nim_positions(neural, seed=seed, return_rows=return_rows)
    methods = ("random", "search", "neural", "hybrid")
    runs = {}
    family_offset = FAMILIES.index(family) * 1000
    for method in methods:
        rows = []
        for episode in range(episodes):
            # Identical scenario seeds and episode RNG seeds pair each policy's result.
            problem_seed = seed + family_offset + episode * 31
            problem = make_problem(family, problem_seed, difficulty=difficulty)
            rows.append(_episode(
                problem, family, method, neural, problem_seed + 50000, simulations, horizon
            ))
        runs[method] = rows
    runs["search_only"] = runs.pop("search")
    runs["neural_only"] = runs.pop("neural")
    results = {method: _summarize(rows) for method, rows in runs.items()}
    results["paired_comparisons"] = {
        "hybrid_minus_search": _paired_summary(runs["hybrid"], runs["search_only"]),
        "neural_minus_random": _paired_summary(runs["neural_only"], runs["random"]),
    }
    return (results, runs) if return_rows else results


def _family_id_from_key(key):
    try:
        _group, features = _split_target_key(key)
    except (TypeError, ValueError):
        return None
    domain = features[16:20]
    if abs(sum(domain) - 1.0) > 1e-5:
        return None
    return max(range(4), key=lambda index: domain[index])


def leave_one_family_out(targets, *, seed, epochs, eval_episodes, simulations, horizon):
    """Train without a whole family and report paired held-out lower confidence bounds."""
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
        comparison = evaluated["paired_comparisons"]["hybrid_minus_search"]
        neural_transfer = evaluated["paired_comparisons"]["neural_minus_random"]
        fold_problem = make_problem(family, seed + 200000 + heldout_id * 1000)
        fallback_used = not fold_model.supports_features(
            fold_problem.features(fold_problem.initial_state())
        )
        records.append({
            "held_out_family": family,
            "training_families": [name for name in FAMILIES if name != family],
            "training_states": fold_metrics["train_states"],
            "validation_states": fold_metrics["validation_states"],
            "validation_scenarios": fold_metrics["validation_scenarios"],
            "neural_validation_mse": fold_metrics["final_validation_mse"],
            "held_out_metrics": evaluated,
            "hybrid_falls_back_without_held_out_training": fallback_used,
            "neural_only_minus_random_return": neural_transfer["mean_delta"],
            "neural_only_minus_random_ci95_lower": neural_transfer["ci95_lower"],
            "neural_only_minus_random_ci95_upper": neural_transfer["ci95_upper"],
            "hybrid_minus_search_return": comparison["mean_delta"],
            "hybrid_minus_search_ci95_lower": comparison["ci95_lower"],
            "hybrid_minus_search_ci95_upper": comparison["ci95_upper"],
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=410)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--fold-epochs", type=int, default=10)
    parser.add_argument("--episodes-per-family", type=int, default=80)
    parser.add_argument("--target-simulations", type=int, default=32)
    parser.add_argument("--eval-simulations", type=int, default=64)
    parser.add_argument("--target-horizon", type=int, default=10)
    parser.add_argument("--eval-horizon", type=int, default=12)
    parser.add_argument("--fold-eval-episodes", type=int, default=40)
    parser.add_argument("--target-scenarios", type=int, default=24)
    parser.add_argument("--target-states", type=int, default=12)
    parser.add_argument("--train-nim-max", type=int, default=60)
    parser.add_argument("--expert-iterations", type=int, default=2)
    parser.add_argument("--champion", default=None, help="last accepted checkpoint for paired comparison")
    parser.add_argument("--out", default="artifacts/strategic_general_neural.json")
    parser.add_argument("--metrics", default="artifacts/general_metrics.json")
    args = parser.parse_args()
    if not 40 <= args.episodes_per_family <= 1000:
        raise SystemExit("promotion evaluation requires at least 40 episodes per non-adversarial family")
    if (not 20 <= args.fold_eval_episodes <= 1000 or not 8 <= args.target_simulations <= 5000
            or not 8 <= args.eval_simulations <= 5000 or not 4 <= args.target_horizon <= 100
            or not 4 <= args.eval_horizon <= 100 or not 1 <= args.epochs <= 300
            or not 1 <= args.fold_epochs <= 100 or not 4 <= args.target_scenarios <= 1000
            or not 4 <= args.target_states <= 60 or not 30 <= args.train_nim_max <= 100
            or not 0 <= args.expert_iterations <= 5):
        raise SystemExit("training budget or search horizon outside configured bounds")

    started = time.time()
    targets, target_summary = collect_search_targets(
        args.seed, args.target_simulations, args.target_horizon,
        scenarios_per_family=args.target_scenarios, states_per_scenario=args.target_states,
        nim_train_max=args.train_nim_max, iteration=0)
    neural, neural_metrics = GeneralNeuralPolicy.distill(
        targets, epochs=args.epochs, seed=args.seed + 7)
    expert_rounds = []
    for iteration in range(1, args.expert_iterations + 1):
        guided_targets, summary = collect_search_targets(
            args.seed + iteration * 700001, args.target_simulations, args.target_horizon,
            scenarios_per_family=args.target_scenarios, states_per_scenario=args.target_states,
            nim_train_max=args.train_nim_max, guide=neural, iteration=iteration)
        targets.update(guided_targets)
        neural, neural_metrics = GeneralNeuralPolicy.distill(
            targets, epochs=args.epochs, seed=args.seed + 7 + iteration * 31)
        expert_rounds.append({
            "iteration": iteration,
            "generated_targets": summary,
            "cumulative_target_states": len(targets),
            "distillation": neural_metrics,
        })
    neural_path = Path(args.out)
    neural.save(neural_path)

    heldout, heldout_rows = {}, {}
    for family in FAMILIES:
        heldout[family], heldout_rows[family] = evaluate_family(
            family, neural, seed=args.seed + 300000,
            episodes=args.episodes_per_family, simulations=args.eval_simulations,
            horizon=args.eval_horizon, return_rows=True)

    # Deliberately shifted holdout: all three planning families are evaluated
    # on harder configurations absent from the standard evaluation seeds.
    stress_heldout, stress_rows = {}, {}
    for family in FAMILIES[:3]:
        stress_heldout[family], stress_rows[family] = evaluate_family(
            family, neural, seed=args.seed + 900000,
            episodes=args.episodes_per_family, simulations=args.eval_simulations,
            horizon=max(args.eval_horizon, 16), return_rows=True, difficulty=2)

    folds = leave_one_family_out(
        targets, seed=args.seed, epochs=args.fold_epochs,
        eval_episodes=args.fold_eval_episodes,
        simulations=max(8, min(args.eval_simulations, 32)), horizon=min(args.eval_horizon, 10))

    hybrid_vs_search = {
        family: scores["paired_comparisons"]["hybrid_minus_search"]
        for family, scores in heldout.items()
    }
    neural_vs_random_ci = {
        family: scores["paired_comparisons"]["neural_minus_random"]
        for family, scores in heldout.items()
    }
    neural_baseline_pass = all(row["mean_delta"] >= 0.0 for row in neural_vs_random_ci.values())
    search_lcb_pass = all(row["ci95_lower"] >= -0.5 for row in hybrid_vs_search.values())
    fold_lcb = min(row["hybrid_minus_search_ci95_lower"] for row in folds)
    fold_lcb_pass = fold_lcb >= -0.5
    stress_hybrid_vs_search = {
        family: scores["paired_comparisons"]["hybrid_minus_search"]
        for family, scores in stress_heldout.items()
    }
    stress_neural_vs_random = {
        family: scores["paired_comparisons"]["neural_minus_random"]
        for family, scores in stress_heldout.items()
    }
    stress_neural_pass = all(row["mean_delta"] >= 0.0
                             for row in stress_neural_vs_random.values())
    stress_search_lcb_pass = all(row["ci95_lower"] >= -1.0
                                 for row in stress_hybrid_vs_search.values())
    stress_metrics_finite = all(
        math.isfinite(float(scores[method]["mean_return"]))
        and 0.0 <= float(scores[method]["success_rate"]) <= 1.0
        for scores in stress_heldout.values()
        for method in METHODS
    )

    champion_comparison = {"available": False, "compatible": False, "families": {}, "pass": True}
    if args.champion and Path(args.champion).is_file():
        champion = GeneralNeuralPolicy.load(args.champion)
        champion_comparison["available"] = True
        champion_comparison["compatible"] = True
        for family in FAMILIES:
            champion_scores, champion_rows = evaluate_family(
                family, champion, seed=args.seed + 300000,
                episodes=args.episodes_per_family, simulations=args.eval_simulations,
                horizon=args.eval_horizon, return_rows=True)
            paired = _paired_summary(
                heldout_rows[family]["hybrid"], champion_rows["hybrid"]
            )
            champion_comparison["families"][family] = paired
        champion_comparison["minimum_paired_ci95_lower"] = min(
            row["ci95_lower"] for row in champion_comparison["families"].values()
        )
        champion_comparison["pass"] = all(
            row["ci95_lower"] >= -0.5 for row in champion_comparison["families"].values()
        )

    metrics_are_finite = all(
        math.isfinite(float(scores[method]["mean_return"]))
        and 0.0 <= float(scores[method]["success_rate"]) <= 1.0
        for scores in heldout.values()
        for method in METHODS
    )
    acceptance = {
        "scenario_disjoint_validation": (
            not neural_metrics["scenario_group_overlap"]
            and neural_metrics["validation_scenarios"] > 0
            and neural_metrics["train_scenarios"] > 0
        ),
        "search_target_validation_mse_improved": bool(neural_metrics["validation_mse_improved"]),
        "neural_only_vs_random_paired_return_deltas": neural_vs_random_ci,
        "neural_only_beats_or_matches_random_on_every_family": neural_baseline_pass,
        "held_out_hybrid_vs_search_paired_deltas": hybrid_vs_search,
        "hybrid_lower_confidence_bound_beats_search_margin": search_lcb_pass,
        "leave_one_family_out_safe_fallback_min_ci95_lower": fold_lcb,
        "leave_one_family_out_safe_fallback_pass": fold_lcb_pass,
        "leave_one_family_out_neural_only_vs_random_paired_return_deltas": {
            row["held_out_family"]: {
                "mean_delta": row["neural_only_minus_random_return"],
                "ci95_lower": row["neural_only_minus_random_ci95_lower"],
                "ci95_upper": row["neural_only_minus_random_ci95_upper"],
            }
            for row in folds
        },
        "leave_one_family_out_neural_only_beats_or_matches_random_on_every_family": all(
            row["neural_only_minus_random_return"] >= 0.0 for row in folds
        ),
        "leave_one_family_out_interpretation": (
            "safe-fallback gate checks hybrid non-regression when the held-out family is unseen; "
            "neural-only versus random is reported separately as a zero-shot transfer diagnostic"
        ),
        "stress_neural_only_vs_random_paired_return_deltas": stress_neural_vs_random,
        "stress_hybrid_vs_search_paired_deltas": stress_hybrid_vs_search,
        "stress_neural_only_beats_or_matches_random_on_every_family": stress_neural_pass,
        "stress_hybrid_lower_confidence_bound_beats_search_margin": stress_search_lcb_pass,
        "stress_metrics_finite": stress_metrics_finite,
        "champion_comparison": champion_comparison,
        "all_metrics_finite": metrics_are_finite,
        "promote_candidate": bool(
            not neural_metrics["scenario_group_overlap"]
            and neural_metrics["validation_mse_improved"]
            and neural_baseline_pass
            and search_lcb_pass
            and fold_lcb_pass
            and stress_neural_pass
            and stress_search_lcb_pass
            and stress_metrics_finite
            and champion_comparison["pass"]
            and metrics_are_finite
        ),
    }
    metrics = {
        "model": "StrategicReasoner-GeneralActionQValue-PUCT-v4",
        "training": {
            "families": list(FAMILIES),
            "initial_target_generation": target_summary,
            "expert_iteration_rounds": expert_rounds,
            "expert_iterations": args.expert_iterations,
            "epochs": args.epochs,
            "neural_distillation": neural_metrics,
            "validation_protocol": "scenario-grouped disjoint split; no scenario may cross train/validation",
        },
        "held_out_scenarios": heldout,
        "stress_held_out_scenarios": stress_heldout,
        "stress_protocol": (
            "difficulty-2 resource demands, deadline-constrained scheduling and 9x9 hazardous routing; "
            "training mixes base and difficulty-1 scenarios; stress evaluation uses separate scenario seeds"
        ),
        "leave_one_family_out": folds,
        "acceptance": acceptance,
        "checkpoint": {"path": str(neural_path), "bytes": neural_path.stat().st_size},
        "runtime_seconds": round(time.time() - started, 2),
        "validation_scope": (
            "paired synthetic resource allocation, job scheduling, weighted route planning, "
            "and 40 unseen adversarial take-away positions; scenario-disjoint validation, "
            "paired 95% confidence intervals and leave-one-family-out transfer. Not real-world "
            "or general-intelligence evidence."
        ),
    }
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
                            encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))
    if not acceptance["scenario_disjoint_validation"]:
        raise SystemExit("scenario leakage detected or validation groups are empty; candidate rejected")
    if not acceptance["all_metrics_finite"]:
        raise SystemExit("held-out evaluation contained invalid metrics; candidate rejected")
    if not acceptance["promote_candidate"]:
        raise SystemExit("candidate failed paired confidence-bound or champion promotion gates")


if __name__ == "__main__":
    main()
