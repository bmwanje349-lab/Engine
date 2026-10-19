"""Small JSON CLI suitable for Termux/Python deployment on Android."""
import argparse
import json
import sys
from .gridworld import make_world
from .learning import QLearner
from .neural import NeuralPolicy
from .search import plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Strategic Reasoning AI planner demo")
    parser.add_argument("--model", default="artifacts/strategic_policy.json")
    parser.add_argument("--neural-model", default="artifacts/strategic_neural.json")
    parser.add_argument("--seed", type=int, default=410)
    parser.add_argument("--simulations", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()
    try:
        learner = QLearner.load(args.model)
        neural = NeuralPolicy.load(args.neural_model)
        world = make_world(args.seed)
        state = world.initial_state()
        result = plan(world, state, simulations=args.simulations, horizon=args.horizon,
                      seed=args.seed, top_k=3,
                      action_prior=lambda s, a: 0.5 * learner.prior(world, s, a) + 0.5 * neural.prior(world, s, a),
                      prior_weight=0.12)
        print(json.dumps({"engine": "Strategic Reasoning AI", "state": state.__dict__,
                          "world": {"width": world.width, "height": world.height,
                                    "start": world.start, "key": world.key_pos, "goal": world.goal,
                                    "walls": sorted(world.walls), "hazards": sorted(world.hazards)},
                          "decision": result}, indent=2))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
