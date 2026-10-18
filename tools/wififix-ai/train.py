#!/usr/bin/env python3
"""Reproducible WiFiFix v2 training with independent synthetic validation."""
import argparse
import json
import time
from pathlib import Path
import sys
import numpy as np

from wififix_ai.simulator import generate_dataset
from wififix_ai.network import (WiFiFixBrain, MODEL_NAME, MIN_MODEL_BYTES,
                                ARCHITECTURE_VERSION, HIDDEN_SIZES, MIN_PARAMETER_COUNT)
from wififix_ai.schema import CAUSES, ACTIONS, N_FEATURES, FEATURE_NAMES, encode_observation


def accuracy(a, b):
    return float(np.mean(np.asarray(a) == np.asarray(b)))


MIN_CAUSE_ACCURACY = 0.65
MIN_ACTION_ACCURACY = 0.58
MIN_CAUSE_BALANCED_ACCURACY = 0.58
MIN_ACTION_BALANCED_ACCURACY = 0.52


def per_class_recall(actual, predicted, class_names):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    recalls = {}
    for index, name in enumerate(class_names):
        selected = actual == index
        recalls[name] = float(np.mean(predicted[selected] == index)) if selected.any() else 0.0
    return recalls


def balanced_accuracy(actual, predicted, class_names):
    recalls = per_class_recall(actual, predicted, class_names)
    return float(np.mean(list(recalls.values()))) if recalls else 0.0


def acceptance_failures(metrics):
    """Return deterministic quality-gate failures; epoch-to-epoch loss is diagnostic only."""
    failures = []
    if not metrics.get("minimum_20_mb_met", False):
        failures.append("trained model artifact is below the 20 MB requirement")
    validation = metrics.get("final_synthetic_validation", {})
    thresholds = (
        ("cause_accuracy", MIN_CAUSE_ACCURACY),
        ("action_accuracy", MIN_ACTION_ACCURACY),
        ("cause_balanced_accuracy", MIN_CAUSE_BALANCED_ACCURACY),
        ("action_balanced_accuracy", MIN_ACTION_BALANCED_ACCURACY),
    )
    for key, minimum in thresholds:
        value = validation.get(key)
        if value is None or not np.isfinite(value) or value < minimum:
            failures.append(f"{key} below synthetic acceptance floor {minimum:.2f}")
    return failures


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
    # Average over three independent simulator seeds so a single lucky or unlucky
    # synthetic draw is less likely to decide whether CI accepts the model.
    validation_sets = [
        generate_dataset(validation_count, a.seed + offset)
        for offset in (1_000_003, 2_000_033, 3_000_091)
    ]
    Xv = np.concatenate([part[0] for part in validation_sets], axis=0)
    ycv = np.concatenate([part[1] for part in validation_sets], axis=0)
    yav = np.concatenate([part[2] for part in validation_sets], axis=0)
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
            print("Ignoring incompatible cached checkpoint and training a fresh v3 model: " + rejected_checkpoint,
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
    cause_predictions = pc.argmax(1)
    action_predictions = pa.argmax(1)
    cause_recalls = per_class_recall(ycv, cause_predictions, CAUSES)
    action_recalls = per_class_recall(yav, action_predictions, ACTIONS)
    final_metrics = {
        "cause_accuracy": accuracy(cause_predictions, ycv),
        "action_accuracy": accuracy(action_predictions, yav),
        "cause_balanced_accuracy": float(np.mean(list(cause_recalls.values()))),
        "action_balanced_accuracy": float(np.mean(list(action_recalls.values()))),
        "cause_class_recall": cause_recalls,
        "action_class_recall": action_recalls,
        "cause_cross_entropy": float(-np.log(np.maximum(pc[np.arange(len(Xv)), ycv], 1e-8)).mean()),
        "action_cross_entropy": float(-np.log(np.maximum(pa[np.arange(len(Xv)), yav], 1e-8)).mean()),
    }
    metrics = {
        "model": MODEL_NAME,
        "architecture_version": ARCHITECTURE_VERSION,
        "architecture": {"input_features": N_FEATURES, "hidden_layers": list(HIDDEN_SIZES),
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
        "validation_simulator_seeds": [a.seed + offset for offset in (1_000_003, 2_000_033, 3_000_091)],
        "acceptance_thresholds": {
            "cause_accuracy": MIN_CAUSE_ACCURACY,
            "action_accuracy": MIN_ACTION_ACCURACY,
            "cause_balanced_accuracy": MIN_CAUSE_BALANCED_ACCURACY,
            "action_balanced_accuracy": MIN_ACTION_BALANCED_ACCURACY,
        },
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
        "minimum_parameter_count": MIN_PARAMETER_COUNT,
        "trained_from_scratch": not resumed,
        "pretrained_model_loaded": False,
        "features": list(FEATURE_NAMES),
        "scope": metrics["validation_scope"],
    }
    Path("artifacts/brain_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    failures = acceptance_failures(metrics)
    if failures:
        raise SystemExit("synthetic validation acceptance failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
