# WiFiFix AI Engine

WiFiFix AI is an experimental Wi-Fi diagnostic adviser. The original adaptive Termux recovery script remains a separate safety layer; the neural adviser never directly toggles Wi-Fi.

## What changed in v2

- **Substantive model size:** two trainable hidden layers (4096, 1536) and independent fault-cause/action heads, about 6.45 million trainable float32 parameters. The uncompressed parameter checkpoint is approximately 25.8 MB decimal. It is not padded or filled with duplicate data.
- **Chess-style diagnostic look-ahead:** bounded expectimax-style search examines up to three tests ahead. Each test branches into possible outcomes, updates the fault-cause probabilities with Bayes' rule, and searches for the best follow-up test for each branch.
- **Three different candidate routes:** the adviser returns three different first-test choices with contingent branches, rather than treating one fixed troubleshooting sequence as correct for every case.
- **Independent synthetic validation:** metrics are calculated on a separate simulator seed, not on the training rows.
- **Optional verified field training:** privacy-scrubbed and human-verified JSONL records can be used as additional training examples. They are not described as a real-world validation set.
- **Conservative operation:** recommendations are advisory; no automatic Wi-Fi control is performed by the model.

## Run locally

Use Python 3.11+:

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python train.py --samples 16000 --epochs 8 --seed 410
    python advisor.py --model artifacts/brain.npz

The trained model and metrics are written to the artifacts directory. A model-size gate rejects a saved checkpoint below 20,000,000 bytes. The bytes come from trainable weights stored as float32; file padding is not used.

To inspect one observation:

    echo '{"connected":1,"rssi_dbm":-55,"gateway_rtt_ms":5,"gateway_loss_pct":0,"internet_rtt_ms":28,"internet_loss_pct":0,"dns_ok":1,"download_mbps":40,"baseline_mbps":45,"sample_count":15,"reconnect_success_rate":0.75,"measurement_reliability":1}' | python advisor.py --model artifacts/brain.npz

The JSON includes candidate_diagnostic_paths. Each path includes possible outcomes, estimated outcome probabilities, updated cause hypotheses, and the next test for each sufficiently likely branch. These are planning estimates, not promises that a real network will follow a branch exactly.

## Training and evidence limits

The neural network is initialized from random weights; no pretrained model or pretrained parameters are loaded. Synthetic examples let the code be tested repeatably, but accuracy on synthetic data only measures how well it learns the simulator's assumptions. It is not proof of diagnosis accuracy on real routers.

For optional field training, create data/real_training.jsonl using one JSON object per line with these fields:

- observation: measurements accepted by encode_observation
- fault_label: one of the fault causes in schema.py
- action_label: one of the advisory actions in schema.py
- review_status: must be verified
- privacy_scrubbed: must be the boolean true

Only add examples with a reliable human-verified label and remove SSIDs, passwords, IP/MAC addresses, tokens, exact locations, and unrelated identifying telemetry. The .example file is illustrative only and is deliberately rejected by the field loader. Field examples are used for training, but the current validation report is still synthetic; real-field accuracy stays unknown until an independently held-out, human-verified field set is evaluated.

GitHub Actions runs tests, training, independent synthetic checks, inference checks, model-size checks, and multi-path planning checks. Successful runs publish the model and metrics as a workflow artifact. Compatible v2 weights may be resumed from the Actions cache; old v1 checkpoints are discarded rather than silently loaded.

## Scope and safety

The model ranks likely causes and advisory actions; the planner compares possible diagnostic paths. Neither can prove an internet provider or router is at fault without reliable measurements. The existing recovery logic remains authoritative. Weak confidence, disconnected Wi-Fi, incomplete observations, or conflicting evidence should lead to gathering more evidence rather than an automatic radio change.
