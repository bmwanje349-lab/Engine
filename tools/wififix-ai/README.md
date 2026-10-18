# WiFiFix AI Engine

An original, compact neural network trained from random initialization to classify Wi-Fi fault patterns and rank advisory actions. No pretrained model or pretrained weights are downloaded or loaded.

## Run
Install Python 3.11+ and NumPy, then run:

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python train.py --samples 24000 --epochs 30 --seed 410
    python advisor.py --model artifacts/brain.npz

Send one JSON observation to advisor.py through standard input or pass --input-json. Example:

    echo '{"connected":1,"rssi_dbm":-55,"gateway_rtt_ms":5,"gateway_loss_pct":0,"internet_rtt_ms":28,"internet_loss_pct":0,"dns_ok":1,"download_mbps":40,"baseline_mbps":45,"sample_count":15,"reconnect_success_rate":0.75,"measurement_reliability":1}' | python advisor.py --model artifacts/brain.npz

## Model and learning
The feature encoder, synthetic Wi-Fi condition simulator, multi-head neural network, backpropagation, training loop, and inference are implemented in this repository. The first training data are generated from a deterministic synthetic simulator. High accuracy on this simulator means the model learned its generated patterns; it is not evidence that real Wi-Fi faults are diagnosed at the same accuracy.

The compact network should not be padded to reach an 18 MB file size. Size is not a quality metric. Future architecture growth must be justified by held-out performance and real-device training data.

GitHub Actions tests the code, trains the model, verifies the artifact, and publishes the brain and metrics as a workflow artifact. Hosted runners cannot directly observe the user's phone. Scheduled training uses synthetic examples unless privacy-scrubbed, verified field labels are supplied.

## Safety
The AI only advises. Weak confidence, missing measurements, a disconnected network, or conflicting evidence produces GATHER_MORE_EVIDENCE. The AI is not permitted to switch the radio. Existing recovery logic remains in control until real-world evaluation supports a deliberate integration.

Never upload SSIDs, IP or MAC addresses, Wi-Fi passwords, tokens, or other identifying telemetry. Keep only the measurements needed for diagnosis.