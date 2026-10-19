# Engine

WiFiFix AI source, tests, and GitHub Actions training pipeline live under tools/wififix-ai/.

A separate Strategic Reasoning AI prototype now lives under tools/strategic-ai/. It combines bounded Monte Carlo tree search, reinforcement learning, and a compact neural Q-value evaluator; its CI workflow trains and publishes Android-deployable JSON checkpoints.

WiFiFix remains a separate neural diagnostic prototype for Wi-Fi troubleshooting. The adaptive Termux recovery script remains its authoritative safety layer; the neural adviser does not directly toggle Wi-Fi.

Read tools/wififix-ai/README.md and tools/wififix-ai/STATUS.md for WiFiFix scope. Read tools/strategic-ai/README.md for the reasoning engine, training, benchmark limits, and Android/Termux deployment steps.