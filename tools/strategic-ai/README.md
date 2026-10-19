# Strategic Reasoning AI (experimental v3)

A separate planning and reinforcement-learning engine inside the Engine repository. It does not replace WiFiFix AI. The goal is to explore alternatives, score their downstream consequences, learn from stronger search targets, and promote checkpoints only when reproducible held-out evaluations pass.

## Planning and learning

- **Bounded Monte Carlo tree search (MCTS):** stochastic transition branches, UCT, discounted returns, rollouts, leaf evaluation, and ranked alternative first actions. Search remains the decision-maker; learned scores are small advisory priors.
- **Reinforcement learning:** tabular Q-learning provides a compact starting policy for the original stochastic grid world.
- **Search-distilled neural evaluator:** the grid evaluator is now trained primarily on independent MCTS action-return targets. A small sampled Q-table replay set adds state coverage, but the neural model is not merely trained to copy the whole Q-table.
- **Cross-domain evaluator:** `strategic_ai/benchmarks.py` defines four controlled families: diminishing-return resource allocation, dependency/deadline scheduling, weighted hazardous route planning, and a two-player take-away game. The first three use single-agent MCTS; the adversarial game uses minimax/negamax targets rather than incorrectly treating the opponent as a cooperative planner.
- **Comparison and regression gates:** training records search-only, neural-only, hybrid, and random baselines for held-out scenarios. It also runs leave-one-family-out folds in which the evaluator is trained without the test family. A candidate is rejected if search-target validation loss does not improve or if hybrid decision quality regresses beyond the configured margin.
- **Android/Termux-friendly inference:** exported evaluator weights are JSON; inference uses Python's standard library and needs no NumPy, server, or network access. Training requires NumPy and is intended for free GitHub Actions runners. The included CLI returns decisions for each family.

These are small synthetic benchmarks for controlled testing. They are not real-world task data and do not establish general intelligence.

## Train and test

From the `tools/strategic-ai` directory (Python 3.11+):

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python train.py --episodes 500 --eval-episodes 20 --simulations 96 --horizon 18
    python train_general.py --epochs 30 --fold-epochs 10 --episodes-per-family 6

The normal GitHub Actions workflow runs unit tests, trains the grid Q-table, distills its neural evaluator from independent MCTS targets, then trains three cross-domain candidates with independent seeds (410, 777, and 2026). Every candidate must pass neural-vs-random comparisons on all four families, seeded held-out scenarios, validation-loss improvement, and four leave-one-family-out folds. The workflow also launches the inference CLI against all four families. It uploads the accepted checkpoints and metrics as a `strategic-ai-v3-<run-number>` artifact for 30 days. A failed training, regression, or CLI smoke-test gate intentionally prevents artifact publication.

## Compare inference methods

After the general checkpoint exists:

    python -m strategic_ai.general_cli --model artifacts/strategic_general_neural.json --family resource_allocation --mode hybrid
    python -m strategic_ai.general_cli --model artifacts/strategic_general_neural.json --family scheduling --mode search
    python -m strategic_ai.general_cli --model artifacts/strategic_general_neural.json --family route_planning --mode neural
    python -m strategic_ai.general_cli --model artifacts/strategic_general_neural.json --family adversarial --mode search

Modes are `search`, `neural`, and `hybrid`. The adversarial search-only mode uses exact minimax for this bounded game; the hybrid uses bounded neural-guided negamax. Search budgets and horizon limits are checked to avoid unbounded CPU work.

## Android / Termux

Install Python in Termux, download the latest successful workflow artifact from GitHub Actions, and extract its JSON files into `tools/strategic-ai/artifacts/`. Then, from `tools/strategic-ai`, run a command from the section above. For a small phone, start with fewer simulations (for example, 64) and a horizon of 10–12. The neural-only mode is cheaper; hybrid mode spends CPU on search. Training stays on GitHub Actions, and no paid API is required.

## Reading the metrics honestly

- `artifacts/metrics.json` measures the original grid-world policy.
- `artifacts/general_metrics.json` and the two `general_metrics_seed*.json` files report the three independent training runs. They include random, search-only, neural-only, and hybrid results per family, target-state counts, validation loss, and four leave-one-family-out folds.
- Success rate and mean return describe only these seeded synthetic problems. The adversarial benchmark additionally reports exact minimax regret on held-out pile sizes.
- A checkpoint is kept only if the MCTS-target validation loss improves, neural-only return is at least as good as random on every family, hybrid search avoids regressions on held-out families, and all three independently seeded candidates pass. A family not represented in the checkpoint's training domains is marked unsupported; hybrid mode falls back to search instead of trusting an out-of-domain neural prior. Model size alone is never treated as evidence of intelligence.

Search quality still depends on the transition model, reward design, state features, rollout budget, and evaluation coverage. The engine explores multiple candidate futures, but it is an experimental planner—not a superintelligence, a pretrained general-reasoning model, or a substitute for verifying high-stakes decisions.
