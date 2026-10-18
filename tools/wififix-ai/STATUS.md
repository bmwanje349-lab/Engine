# Status: bootstrap prototype

Implemented in source: randomized neural model and custom backpropagation, 32-feature schema, synthetic condition simulator, training CLI, inference adviser, conservative safety gate, unit tests, and a repeatable GitHub Actions training workflow.

The earlier local prototype passed five unit tests and achieved approximately 99% label accuracy against independent synthetic examples. These are synthetic-generator results only. Real-world Wi-Fi accuracy is unknown and needs anonymized, verified field measurements.

The trained prototype checkpoint should be recreated by the GitHub Actions workflow and downloaded as an artifact. It will be much smaller than 18 MB; the model is not padded. The requested size is a ceiling/target to revisit only if additional data and architecture justify it.

Pending: live GitHub Actions verification, real-device evaluation, model calibration, privacy-reviewed field data, and any controlled integration with the preserved existing WiFiFix recovery logic. Automatic AI-triggered Wi-Fi toggling remains disabled.