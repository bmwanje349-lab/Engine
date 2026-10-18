# WiFiFix AI Engine

WiFiFix AI is an experimental Wi-Fi diagnostic adviser. The authoritative adaptive Termux recovery engine is included at `recovery/wififix_pro_v2.sh`; it remains separate from the neural adviser, which never directly toggles Wi-Fi.

## What changed in v3

- **Substantive model size:** hidden layers (3072, 1664) with independent fault-cause/action heads, about 5.24 million trainable float32 parameters and approximately 20.96 MB decimal of actual weights. This stays close to the hard 20 MB minimum without padding or duplicate data.
- **Chess-style diagnostic look-ahead:** bounded expectimax-style search examines up to three tests ahead. Each test branches into possible outcomes, updates the fault-cause probabilities with Bayes' rule, and searches for the best follow-up test for each branch.
- **Alternative diagnostic routes:** the adviser returns up to three distinct first-test choices with contingent branches, excluding tests already supported by adequate measurements rather than repeating a fixed troubleshooting sequence.
- **Less noisy synthetic validation:** final scores pool three independent simulator seeds and report per-class recall plus class-balanced accuracy. CI gates require at least 0.90 cause accuracy, 0.88 action accuracy, 0.85 cause-balanced accuracy, and 0.82 action-balanced accuracy. These strict floors still measure simulator performance only, not real-world diagnosis.
- **Optional verified field training:** privacy-scrubbed and human-verified JSONL records can be used as additional training examples. They are not described as a real-world validation set.
- **Conservative operation:** recommendations are advisory; no automatic Wi-Fi control is performed by the model.

## Run locally

Use Python 3.11+:

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python train.py --samples 16000 --epochs 8 --seed 410
    python advisor.py --model artifacts/brain.npz

The trained model and metrics are written to the artifacts directory. A model-size gate rejects a saved checkpoint below 20,000,000 bytes. The minimum is met by actual float32 trainable weights; file padding is not used. Architecture v3 intentionally invalidates older cached v2 dimensions, which are rejected and retrained rather than silently reused.

To inspect one observation:

    echo '{"connected":1,"rssi_dbm":-55,"gateway_rtt_ms":5,"gateway_loss_pct":0,"internet_rtt_ms":28,"internet_loss_pct":0,"dns_ok":1,"download_mbps":40,"baseline_mbps":45,"sample_count":15,"reconnect_success_rate":0.75,"measurement_reliability":1}' | python advisor.py --model artifacts/brain.npz

The JSON includes candidate_diagnostic_paths. Each path includes possible outcomes, estimated outcome probabilities, updated cause hypotheses, and the next test for each sufficiently likely branch. These are planning estimates, not promises that a real network will follow a branch exactly.

## Authoritative Termux recovery layer

`recovery/wififix_pro_v2.sh` is the checked-in recovery script. It retains saved network profiles, waits for repeated measured failures before considering recovery, backs off after unsuccessful attempts, and does not toggle Wi-Fi when connection state cannot be confirmed. It does not use predictions from the neural model to control the radio.

In Termux, install its dependencies and copy the script to a convenient location, for example:

    pkg install termux-api jq curl iproute2
    cp recovery/wififix_pro_v2.sh ~/wififix_pro_v2.sh
    chmod 700 ~/wififix_pro_v2.sh
    bash ~/wififix_pro_v2.sh diagnose
    bash ~/wififix_pro_v2.sh start
    bash ~/wififix_pro_v2.sh status

Automatic Wi-Fi switching still depends on Android permissions and an available control method; Termux:API alone may not be permitted to toggle Wi-Fi on every Android version. Use `pause` to enforce monitoring-only behavior, and `stop` to stop the monitor.

## Training and evidence limits

The neural network is initialized from random weights; no pretrained model or pretrained parameters are loaded. Synthetic examples let the code be tested repeatably, but accuracy on synthetic data only measures how well it learns the simulator's assumptions. It is not proof of diagnosis accuracy on real routers.

For optional field training, create data/real_training.jsonl using one JSON object per line with these fields:

- observation: measurements accepted by encode_observation
- fault_label: one of the fault causes in schema.py
- action_label: one of the advisory actions in schema.py
- review_status: must be verified
- privacy_scrubbed: must be the boolean true

Only add examples with a reliable human-verified label and remove SSIDs, passwords, IP/MAC addresses, tokens, exact locations, and unrelated identifying telemetry. The .example file is illustrative only and is deliberately rejected by the field loader. Field examples are used for training, but the current validation report is still synthetic; real-field accuracy stays unknown until an independently held-out, human-verified field set is evaluated.

GitHub Actions runs unit tests, recovery-script syntax checks, training, multi-seed synthetic checks, inference checks, model-size checks, and evidence-aware multi-path planning checks. Successful runs publish the model and metrics as a workflow artifact. Compatible v3 weights may be resumed from the Actions cache; older checkpoints are discarded rather than silently loaded.

## Scope and safety

The model ranks likely causes and advisory actions; the planner compares possible diagnostic paths. Neither can prove an internet provider or router is at fault without reliable measurements. The existing recovery logic remains authoritative. Weak confidence, disconnected Wi-Fi, incomplete observations, or conflicting evidence should lead to gathering more evidence rather than an automatic radio change.
