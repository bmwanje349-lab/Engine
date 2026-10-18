#!/usr/bin/env python3
"""Reproducible WiFiFix v2 training with independent synthetic validation."""
import argparse
import json
import time
from pathlib import Path
import sys
import numpy as np

from wififix_ai.simulator import generate_dataset
from wififix_ai.network import WiFiFixBrain, MODEL_NAME, MIN_MODEL_BYTES
from wififix_ai.schema import CAUSES, ACTIONS, N_FEATURES, FEATURE_NAMES, encode_observation


def accuracy(a, b):
    return float(np.mean(np.asarray(a) == np.asarray(b)))


def load_verified_field_examples(path):
    """Load explicitly privacy-scrubbed, human-verified JSONL training records.

    Synthetic examples and unreviewed telemetry are rejected. These examples
    join the training set only; reported validation metrics remain synthetic
    and never pretend to be real-field accuracy.
    """
    path = Path(path)
    if not path.exists():
        return (
            np.empty((0, N_FEATURES), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    features, causes, actions = [], [], []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if row.get("review_status") != "verified" or row.get("privacy_scrubbed") is not True:
                raise SystemExit(
                    f"Rejected {path}:{line_number}: field data must be human-verified "
                    "and explicitly privacy-scrubbed"
                )
            observation = row.get("observation")
            cause = row.get("fault_label")
            action = row.get("action_label")
            if not isinstance(observation, dict) or cause not in CAUSES or action not in ACTIONS:
                raise SystemExit(f"Invalid observation or label in {path}:{line_number}")
            try:
                features.append(encode_observation(observation))
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"Invalid features in {path}:{line_number}: {exc}") from exc
            causes.append(CAUSES.index(cause))
            actions.append(ACTIONS.index(action))
    return (
        np.asarray(features, dtype=np.float32).reshape((-1, N_FEATURES)),
        np.asarray(causes, dtype=np.int64),
        np.asarray(actions, dtype=np.int64),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=16000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=410)
    p.add_argument("--out", default="artifacts/brain.npz")
    p.add_argument("--metrics", default="artifacts/metrics.json")
    p.add_argument("--resume", default="")
    p.add_argument("--field-data", default="data/real_training.jsonl")
    a = p.parse_args()
    if not 1000 <= a.samples <= 250000 or not 1 <= a.epochs <= 30:
        raise SystemExit("sample or epoch limit rejected")

    started = time.time()
    X, yc, ya = generate_dataset(a.samples, a.seed)
    validation_count = max(2000, min(8000, a.samples // 4))
    Xv, ycv, yav = generate_dataset(validation_count, a.seed + 1_000_003)
    Xf, ycf, yaf = load_verified_field_examples(a.field_data)
    if len(Xf):
        X = np.concatenate((X, Xf), axis=0)
        yc = np.concatenate((yc, ycf), axis=0)
        ya = np.concatenate((ya, yaf), axis=0)
    order = np.random.default_rng(a.seed + 1).permutation(len(X))
    X, yc, ya = X[order], yc[order], ya[order]

    resumed = False
    rejected_checkpoint = ""
    if a.resume and Path(a.resume).exists():
        try:
            model = WiFiFixBrain.load(a.resume)
            resumed = True
        except (ValueError, OSError, KeyError) as exc:
            rejected_checkpoint = str(exc)
            print("Ignoring incompatible cached checkpoint and training a fresh v2 model: " + rejected_checkpoint,
                  file=sys.stderr)
            model = WiFiFixBrain(a.seed)
    else:
        model = WiFiFixBrain(a.seed)

    before_c, before_a = model.probabilities(Xv)
    before = {
        "cause_accuracy": accuracy(before_c.argmax(1), ycv),
        "action_accuracy": accuracy(before_a.argmax(1), yav),
    }
    losses = model.fit(X, yc, ya, epochs=a.epochs, seed=a.seed + 2)
    pc, pa = model.probabilities(Xv)
    out = Path(a.out)
    model.save(out)
    final_metrics = {
        "cause_accuracy": accuracy(pc.argmax(1), ycv),
        "action_accuracy": accuracy(pa.argmax(1), yav),
        "cause_cross_entropy": float(-np.log(np.maximum(pc[np.arange(len(Xv)), ycv], 1e-8)).mean()),
        "action_cross_entropy": float(-np.log(np.maximum(pa[np.arange(len(Xv)), yav], 1e-8)).mean()),
    }
    metrics = {
        "model": MODEL_NAME,
        "architecture_version": 2,
        "architecture": {"input_features": N_FEATURES, "hidden_layers": [4096, 1536],
                          "prediction_heads": {"fault_cause": len(CAUSES), "advisory_action": len(ACTIONS)}},
        "parameter_count": model.parameter_count,
        "trainable_parameter_bytes": model.trainable_bytes,
        "weights_origin": "compatible-own-checkpoint-resume" if resumed else "random-initialization",
        "pretrained_weights_used": False,
        "rejected_incompatible_checkpoint": rejected_checkpoint or None,
        "feature_count": N_FEATURES,
        "cause_classes": list(CAUSES),
        "action_classes": list(ACTIONS),
        "seed": a.seed,
        "synthetic_training_samples": a.samples,
        "verified_field_examples_used": int(len(Xf)),
        "field_examples_validated": 0,
        "training_samples_total": len(X),
        "validation_samples": len(Xv),
        "epochs": a.epochs,
        "initial_synthetic_validation": before,
        "final_synthetic_validation": final_metrics,
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "loss_improved": losses[-1] < losses[0],
        "model_bytes": out.stat().st_size,
        "model_size_mb_decimal": round(out.stat().st_size / 1_000_000, 4),
        "minimum_20_mb_met": out.stat().st_size >= MIN_MODEL_BYTES,
        "elapsed_seconds": round(time.time() - started, 2),
        "validation_scope": "independent synthetic simulator only; not real-world Wi-Fi validation",
        "real_field_accuracy": None,
        "automatic_network_control": "disabled; recommendations are advisory",
    }
    Path(a.metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(a.metrics).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "format": "uncompressed NumPy NPZ",
        "model_file": str(out),
        "model_bytes": out.stat().st_size,
        "model_size_mb_decimal": round(out.stat().st_size / 1_000_000, 4),
        "minimum_file_bytes": MIN_MODEL_BYTES,
        "minimum_size_met": out.stat().st_size >= MIN_MODEL_BYTES,
        "parameter_count": model.parameter_count,
        "trainable_parameter_bytes": model.trainable_bytes,
        "trained_from_scratch": not resumed,
        "pretrained_model_loaded": False,
        "features": list(FEATURE_NAMES),
        "scope": metrics["validation_scope"],
    }
    Path("artifacts/brain_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    if not metrics["minimum_20_mb_met"]:
        raise SystemExit("trained model artifact is below the 20 MB requirement")
    if not metrics["loss_improved"]:
        raise SystemExit("training loss did not improve")
    if final_metrics["cause_accuracy"] < .55 or final_metrics["action_accuracy"] < .45:
        raise SystemExit("independent synthetic validation fell below bootstrap acceptance thresholds")


if __name__ == "__main__":
    main()
