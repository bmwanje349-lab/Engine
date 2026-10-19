# Strategic Reasoning AI (experimental v5)

A research prototype inside the Engine repository, separate from WiFiFix AI. It explores alternatives, compares future outcomes, and promotes a checkpoint only when explicit held-out quality and regression gates pass. It is not a claim of superintelligence or real-world reliability.

## Planning and learning

- **PUCT planning:** the tree policy balances observed action returns against a normalized learned prior. Search has bounded simulations and horizon, discounted backups, alternative first actions, and an optional learned value estimate at search leaves.
- **Reusable search:** each planning episode can retain a bounded transposition table; repeated Markov states share nodes. A caller can promote the observed next state to become the next root. Root Dirichlet noise supports exploration while training search-guided targets.
- **Action-conditioned neural evaluator:** the Q head evaluates an explicit (state, action) pair through domain-aware action features. Allocation uses project-specific actions, scheduling uses job-specific actions, route planning uses directions, and the adversarial game uses explicit take-one-to-five actions. A separate value head predicts state value for PUCT leaf evaluation.
- **Expert iteration:** search generates action-return targets, the model is trained, then the model guides new PUCT searches that generate more targets. The network is retrained on cumulative data rather than being distilled only once.
- **Four controlled benchmark families:** diminishing-return resource allocation, dependency/deadline scheduling, weighted hazardous route planning, and an adversarial take-away game. The first three use model-based planning; the adversarial family uses minimax/negamax rather than treating an opponent as a cooperative agent.
- **Scenario-disjoint validation:** target rows carry scenario identifiers, so all states from one scenario stay in either training or validation. Q and value targets are normalized separately within each domain and inverse-scaled at inference; minibatches oversample smaller domains so the adversarial task is not drowned by the planning data. CI reports finite-sample Student-t confidence intervals and gates promotion on lower confidence bounds, leave-one-family-out safe-fallback tests, comparisons against random/search baselines, and—when one exists—a same-scenario paired comparison against the last released champion. The report separately measures neural-only versus random on each held-out family to expose zero-shot transfer limits.
- **Standard-library inference:** NumPy is only needed for training. The JSON checkpoint and four-family CLI use the Python standard library, checked in dedicated jobs on Python 3.12 and 3.13 without installing NumPy.

Benchmark scores are limited to these synthetic environments. They do not establish broad general reasoning, real-world task performance, or safety.

## Train and test locally

From the tools/strategic-ai directory on Python 3.11 or newer:

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python train.py --episodes 500 --eval-episodes 80 --simulations 96 --horizon 18
    python train_general.py --epochs 45 --fold-epochs 10 --episodes-per-family 80 --target-scenarios 24 --target-states 12 --expert-iterations 2

For a faster local check, reduce the training episodes and search simulations; retain at least 40 paired episodes per non-adversarial family for a promotion-quality evaluation.

## CI and scale

The strategic-ai workflow pins hosted jobs to ubuntu-24.04, uses current major versions of checkout, setup-python and upload/download-artifact, and cancels stale runs on the same branch. Workflow inputs are passed into scripts through environment variables and validated before training. The three challenger seeds (410, 777 and 2026) train in parallel.

Pull requests run a bounded smoke budget. A merge to main and weekly scheduled training use 480 scenarios per planning family and 12 target states per scenario, followed by two guided expert-iteration rounds; when all three challenger seeds pass the gates, the main-branch run publishes the selected checkpoint, metrics, and SHA-256 manifest as a GitHub Release. The raw target-generation budget is up to 60 times a single 24-scenario × 12-state × 3-family round before duplicate-state effects; the adversarial family also contributes exact minimax-labeled positions. If the runtime on hosted runners exceeds the 60-minute job budget, reduce simulation count or scenarios rather than weakening evaluation gates.

Promotion is deliberately strict: a candidate must improve domain-balanced scenario-held-out Q/value loss, avoid train/validation scenario overlap, meet the predeclared paired lower-bound gate versus search on every family, pass leave-one-family-out hybrid safe-fallback non-regression, beat random with a positive paired 95% lower bound on unseen difficulty-2 resource-allocation, scheduling and route-planning scenarios, and avoid a material paired regression versus the released champion. The leave-one-family-out neural-only versus random measurements are reported separately as zero-shot transfer diagnostics; passing the fallback gate does not prove that the neural network itself generalizes to a domain it never saw. The release job only selects checkpoints whose metrics explicitly mark them accepted.

## Android / Termux

Install Python in Termux, then retrieve the latest released checkpoint and its SHA-256 manifest:

    mkdir -p ~/strategic-ai && cd ~/strategic-ai
    curl -fL -o strategic_general_neural.json https://github.com/bmwanje349-lab/Engine/releases/latest/download/strategic_general_neural.json
    curl -fL -o SHA256SUMS https://github.com/bmwanje349-lab/Engine/releases/latest/download/SHA256SUMS
    sha256sum -c SHA256SUMS

Run from the directory containing the downloaded model:

    python -m strategic_ai.general_cli --model strategic_general_neural.json --family route_planning --mode hybrid --simulations 64 --horizon 10

For those commands, clone or download the tools/strategic-ai source and run the CLI from that directory. Alternatively, set PYTHONPATH to the source directory. No NumPy, server, paid API or network access is needed at inference time after the source and checkpoint are present.

Each accepted release contains the selected champion, individual seed candidates and metrics, the grid-world artifacts, the selection summary, and SHA256SUMS. The manifest hashes every JSON asset in that release (the manifest does not hash itself). Checkpoints can be replaced by a later release; record the chosen release tag if reproducibility matters.

## Reading the evidence honestly

- The artifacts/general_metrics.json file records the selected checkpoint's training and held-out metrics. general_metrics_seed*.json stores each independently trained candidate and its explicit promotion decision.
- champion_selection.json records which accepted seed was selected and the paired confidence-bound score used to choose it.
- metrics.json measures the separate original grid-world policy; it is not the same evaluator as the cross-domain checkpoint.
- A checkpoint is accepted only against these predeclared synthetic tests. Search quality still depends on the transition model, feature representation, reward design, search budget and benchmark coverage. The system explores multiple possible futures but is experimental decision-support code, not a general intelligence or a substitute for independently checking consequential decisions.
