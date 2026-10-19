#!/usr/bin/env python3
"""Train the Strategic Reasoning AI policy prior and benchmark look-ahead search."""
from __future__ import annotations
import argparse
import json
import random
import statistics
import time
from pathlib import Path

from strategic_ai.gridworld import GridWorld, make_world
from strategic_ai.learning import QLearner, encode, state_features
from strategic_ai.neural import NeuralPolicy
from strategic_ai.search import plan


def random_episode(world: GridWorld, seed: int) -> dict:
    rng, state = random.Random(seed), world.initial_state()
    total, success, steps = 0.0, False, 0
    while not world.terminal(state) and steps < world.max_steps:
        legal = list(world.actions(state))
        if not legal:
            break
        state, reward, done, info = world.step(state, rng.choice(legal), rng)
        total += reward
        steps += 1
        success |= bool(info.get("goal_reached"))
        if done:
            break
    return {"return": total, "success": success, "steps": steps}


def collect_search_targets(seed: int, simulations: int, horizon: int) -> tuple[dict, dict]:
    """Generate action-value labels from stronger MCTS look-ahead, not Q-table imitation."""
    targets: dict[str, dict[str, float]] = {}
    visited = 0
    target_simulations = max(32, min(96, simulations))
    target_horizon = max(8, min(14, horizon))
    for map_index in range(12):
        world = make_world(seed + 20000 + map_index * 31)
        rng = random.Random(seed + 80000 + map_index)
        state = world.initial_state()
        for step_index in range(7):
            if world.terminal(state):
                break
            legal = list(world.actions(state))
            if not legal:
                break
            # No learned prior is used to make the teacher independent of its student.
            result = plan(
                world, state, simulations=target_simulations, horizon=target_horizon,
                seed=seed + map_index * 1000 + step_index * 19,
                top_k=len(legal), prior_weight=0.0,
            )
            row = targets.setdefault(encode(state_features(world, state)), {})
            for alternative in result["alternatives"]:
                action = alternative["first_action"]
                value = float(alternative["estimated_return"])
                if action in legal and value == value and abs(value) < 1e6:
                    row[action] = value
            visited += 1
            action = (rng.choice(legal) if rng.random() < 0.20 else result["best_action"])
            if action is None:
                break
            state, _reward, done, _info = world.step(state, action, rng)
            if done:
                break
    targets = {key: row for key, row in targets.items() if row}
    if len(targets) < 20:
        raise RuntimeError(f"MCTS target collection produced too few unique states: {len(targets)}")
    return targets, {
        "teacher": "independent MCTS return estimates",
        "target_simulations": target_simulations,
        "target_horizon": target_horizon,
        "trajectory_states_visited": visited,
        "unique_search_target_states": len(targets),
        "target_action_values": sum(len(row) for row in targets.values()),
    }


def make_distillation_targets(q_table: dict, search_targets: dict, seed: int) -> tuple[dict, int]:
    """Mix broad but sparse Q-table coverage with a majority of stronger search labels."""
    rng = random.Random(seed)
    unlabelled_by_search = [key for key in q_table if key not in search_targets]
    q_sample_count = min(32, max(8, len(unlabelled_by_search) // 30))
    sampled = rng.sample(unlabelled_by_search, min(q_sample_count, len(unlabelled_by_search)))
    combined = {key: dict(q_table[key]) for key in sampled}
    # Stronger search targets overwrite any duplicate state labels and dominate sample count.
    combined.update({key: dict(row) for key, row in search_targets.items()})
    return combined, len(sampled)


def run_mcts_episode(world: GridWorld, model: QLearner, neural: NeuralPolicy, seed: int, simulations: int,
                     horizon: int) -> dict:
    rng, state = random.Random(seed), world.initial_state()
    total, success, steps, planning_nodes = 0.0, False, 0, 0
    while not world.terminal(state) and steps < world.max_steps:
        result = plan(world, state, simulations=simulations, horizon=horizon,
                      seed=rng.randrange(2**31), top_k=3,
                      action_prior=lambda s, a: 0.5 * model.prior(world, s, a) + 0.5 * neural.prior(world, s, a),
                      prior_weight=0.12)
        action = result["best_action"]
        if action is None:
            break
        planning_nodes += result["root_visits"]
        state, reward, done, info = world.step(state, action, rng)
        total += reward
        steps += 1
        success |= bool(info.get("goal_reached"))
        if done:
            break
    return {"return": total, "success": success, "steps": steps, "search_visits": planning_nodes}


def run_neural_episode(world: GridWorld, neural: NeuralPolicy, seed: int) -> dict:
    rng, state = random.Random(seed), world.initial_state()
    total, success, steps = 0.0, False, 0
    while not world.terminal(state) and steps < world.max_steps:
        legal = list(world.actions(state))
        if not legal:
            break
        values = neural.predict_features(state_features(world, state))
        action = max(legal, key=lambda a: values[("UP", "RIGHT", "DOWN", "LEFT", "REST").index(a)])
        state, reward, done, info = world.step(state, action, rng)
        total += reward
        steps += 1
        success |= bool(info.get("goal_reached"))
        if done:
            break
    return {"return": total, "success": success, "steps": steps}


def summarize(rows: list[dict]) -> dict:
    n = max(1, len(rows))
    return {"episodes": len(rows),
            "success_rate": round(sum(bool(x["success"]) for x in rows) / n, 4),
            "mean_return": round(statistics.fmean(float(x["return"]) for x in rows), 5),
            "mean_steps": round(statistics.fmean(float(x["steps"]) for x in rows), 3)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--simulations", type=int, default=96)
    p.add_argument("--horizon", type=int, default=18)
    p.add_argument("--seed", type=int, default=410)
    p.add_argument("--out", default="artifacts/strategic_policy.json")
    p.add_argument("--metrics", default="artifacts/metrics.json")
    args = p.parse_args()
    if not 50 <= args.episodes <= 100000 or not 4 <= args.eval_episodes <= 1000:
        raise SystemExit("episode count outside configured safe bounds")
    if not 8 <= args.simulations <= 5000 or not 4 <= args.horizon <= 100:
        raise SystemExit("search budget outside configured safe bounds")

    started = time.time()
    learner = QLearner(seed=args.seed)
    curriculum = [make_world(args.seed + k) for k in (1, 2, 3, 4, 5)]
    training_returns = []
    for ep in range(args.episodes):
        world = curriculum[ep % len(curriculum)] if ep % 4 else make_world(args.seed + 10000 + ep)
        training_returns.append(learner.train_episode(world))

    search_targets, search_target_metrics = collect_search_targets(args.seed, args.simulations, args.horizon)
    distillation_targets, auxiliary_q_states = make_distillation_targets(
        learner.q, search_targets, args.seed + 31)
    neural, neural_metrics = NeuralPolicy.distill(
        distillation_targets, epochs=35, seed=args.seed + 9)
    neural_metrics.update({
        "teacher_source": "MCTS search-generated returns plus a small auxiliary Q-table replay sample",
        "search_target_states": search_target_metrics["unique_search_target_states"],
        "search_target_action_values": search_target_metrics["target_action_values"],
        "auxiliary_q_states": auxiliary_q_states,
    })
    neural_path = Path("artifacts/strategic_neural.json")
    neural.save(neural_path)

    learned_rows, neural_rows, random_rows, mcts_rows = [], [], [], []
    for i in range(args.eval_episodes):
        world = make_world(args.seed + 300000 + i)
        learned_rows.append(learner.evaluate_episode(world, seed=args.seed + 400000 + i))
        neural_rows.append(run_neural_episode(world, neural, args.seed + 450000 + i))
        random_rows.append(random_episode(world, args.seed + 500000 + i))
    mcts_count = min(args.eval_episodes, 12)
    for i in range(mcts_count):
        world = make_world(args.seed + 300000 + i)
        mcts_rows.append(run_mcts_episode(world, learner, neural, args.seed + 700000 + i,
                                          args.simulations, args.horizon))

    output = Path(args.out)
    learner.save(output)
    metrics = {
        "model": "StrategicReasoner-QTable-v2-search-distilled",
        "training": {"episodes": args.episodes,
                     "mean_return_first_20": round(statistics.fmean(x["return"] for x in training_returns[:min(20, len(training_returns))]), 5),
                     "mean_return_last_20": round(statistics.fmean(x["return"] for x in training_returns[-min(20, len(training_returns)):]), 5),
                     "epsilon_final": round(learner.epsilon, 5), "q_states": len(learner.q),
                     "updates": learner.updates},
        "search_target_generation": search_target_metrics,
        "held_out_random_policy": summarize(random_rows),
        "held_out_learned_policy": summarize(learned_rows),
        "held_out_neural_policy": summarize(neural_rows),
        "held_out_mcts_policy": summarize(mcts_rows),
        "held_out_mcts_comparison_random_policy": summarize(random_rows[:mcts_count]),
        "neural_distillation": neural_metrics,
        "neural_checkpoint": {"path": str(neural_path), "bytes": neural_path.stat().st_size},
        "search": {"simulations_per_decision": args.simulations, "horizon": args.horizon,
                   "mean_root_visits": round(statistics.fmean(x.get("search_visits", 0) / max(1, x["steps"]) for x in mcts_rows), 2)},
        "checkpoint": {"path": str(output), "bytes": output.stat().st_size},
        "runtime_seconds": round(time.time() - started, 2),
        "validation_scope": "synthetic stochastic gridworld; neural teacher labels now come from independent MCTS, not only Q-table values",
        "random_baseline_beaten_on_return": summarize(learned_rows)["mean_return"] > summarize(random_rows)["mean_return"],
        "hybrid_search_beats_random_on_return": summarize(mcts_rows)["mean_return"] > summarize(random_rows[:mcts_count])["mean_return"],
    }
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
