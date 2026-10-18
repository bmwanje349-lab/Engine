# Status: WiFiFix AI v2 candidate

## Implemented in this branch

- Expanded the neural adviser to a two-hidden-layer network (4096 and 1536 units) with fault-cause and action heads.
- Checkpoint stores 6.45 million trainable float32 parameters in an uncompressed NPZ archive. A 20,000,000-byte minimum is enforced by both model save code and the training workflow; no filler bytes or duplicate payloads are added.
- Added an expectimax-style diagnostic planner. It evaluates alternate signal, gateway, internet, DNS, speed, and stability tests, branches on likely outcomes, updates cause beliefs, and returns the top three first-test routes with follow-up tests for each branch.
- Added an independently seeded synthetic validation set, checkpoint compatibility checks, privacy-reviewed field-data loading, and tests for planner diversity and output size.
- Kept the inference layer advisory-only. Wi-Fi radio switching remains disabled.

## Important limitations

- No pretrained model is loaded.
- The model has only synthetic validation unless an operator explicitly supplies privacy-scrubbed, human-verified field examples.
- Field examples added to training do not count as independent validation. Real-world accuracy remains unknown until a separate verified field evaluation is run.
- A checkpoint exceeding 20 MB meets the file-size requirement, not a claim of superhuman or production-grade intelligence.

## Release gate

Pending: GitHub Actions on this branch must pass unit tests, training, independent synthetic checks, checkpoint-size checks, and the branching-planner validation. The trained model and metrics should be taken from the successful Actions artifact, not assumed from source code alone.
