# Strategic Reasoning AI (experimental v1)

A separate planning and reinforcement-learning engine inside the Engine repository. It does not replace WiFiFix AI. This first version focuses on a verifiable core: exploring multiple possible futures, ranking distinct first actions, learning from rewards and failures, and publishing reproducible benchmark checkpoints.

## What is implemented

- Domain-agnostic Monte Carlo tree search (MCTS): bounded simulations, UCT exploration, stochastic transition branches, discounted-return backpropagation, rollouts, and a ranked set of distinct first moves.
- Reinforcement learning: tabular Q-learning with epsilon exploration/decay, bootstrapped updates and JSON checkpoint persistence. Failed or low-value actions lower future estimates; high-return decisions receive stronger values.
- Hybrid reasoning: learned Q-values and a compact neural Q-value approximator act only as bounded priors. Search still explores alternatives and can override stale learned preferences. The network is distilled from the RL table and trained from scratch; it is not a pretrained general reasoning model.
- Reference environment: a stochastic grid-planning challenge with obstacles, a key/goal dependency, energy constraints, hazards and variable maps. This makes progress measurable but is not general intelligence by itself.
- Android-ready interface: standard-library Python inference; no paid API and no third-party runtime dependencies at inference. The CLI emits JSON and can run in Termux. A native Kotlin UI/runtime can be added once the engine has stable benchmark results.
- CI: unit tests, deterministic repeatability, held-out seeded benchmarks, a saved learned policy, metrics and a downloadable Actions artifact.

## Run in GitHub Codespaces, Linux, or Termux

Python 3.11 or newer is recommended. Inference uses only the Python standard library. Training the neural evaluator requires NumPy, installed automatically by the GitHub Actions workflow.

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python train.py --episodes 500 --eval-episodes 20 --simulations 96 --horizon 18
    python -m strategic_ai.cli --model artifacts/strategic_policy.json --neural-model artifacts/strategic_neural.json --simulations 512 --horizon 24

On Android Termux, install Python with pkg install python and get this repository. For lightweight deployment, download strategic_policy.json and strategic_neural.json from the latest successful strategic-ai-v1 GitHub Actions artifact, extract them into tools/strategic-ai/artifacts/, then run the CLI command above. Running inference needs no NumPy, server or network permission; training the neural evaluator does require NumPy. GitHub Actions uses free hosted runners subject to current GitHub usage limits and publishes artifacts for 30 days.

## How to read results

artifacts/metrics.json compares a random policy, the learned Q policy, and MCTS-guided planning on held-out generated maps. MCTS reports success rate, average return, mean steps and decision-search visits; its random baseline uses the same held-out map seeds for a fairer comparison. The artifacts contain the learned Q-table and real trained neural weights, not filler data. Seeded synthetic results are reproducible and should not be described as real-world or general-intelligence performance.

Search scores depend on transition-model quality, reward design, search budget and state representation. A narrow benchmark can be overfit. Before claiming that learning generalizes, expand to different planning domains and use held-out task families, not only new random seeds of one gridworld.

## Roadmap

1. This baseline: testable MCTS + Q-learning + reproducible free CI.
2. Train the neural value/policy evaluator directly against improved-search targets; compare MCTS-only, learned-policy-only and hybrid on multiple held-out task families.
3. Add an Android Kotlin front end and a stable JSON request/response API, while keeping the planner bounded for phone memory, battery and latency.
4. Add more domains (resource allocation, route planning, scheduling and adversarial games), stronger uncertainty calibration, ablations and regression-gated model promotion.
5. Learn from explicitly recorded, non-sensitive feedback. Keep benchmark seeds isolated and never merge unverified user data into evaluation.

## Safety and limits

This is a specialized experimental planner, not a superintelligence or autonomous all-purpose agent. It does not execute external actions; it evaluates simulated decisions only. The reference environment is synthetic. Good benchmark results demonstrate performance on that benchmark, not universal reasoning ability.
