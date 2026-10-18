#!/usr/bin/env python3
"""Bounded, reproducible training from random initialization or our own checkpoint."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from wififix_ai.simulator import generate_dataset
from wififix_ai.network import WiFiFixBrain
from wififix_ai.schema import CAUSES, ACTIONS, N_FEATURES, FEATURE_NAMES

def accuracy(a, b):
    return float(np.mean(np.asarray(a) == np.asarray(b)))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=24000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=410)
    p.add_argument("--out", default="artifacts/brain.npz")
    p.add_argument("--metrics", default="artifacts/metrics.json")
    p.add_argument("--resume", default="")
    a = p.parse_args()
    if not 1000 <= a.samples <= 250000 or not 1 <= a.epochs <= 120:
        raise SystemExit("sample or epoch limit rejected")
    started = time.time()
    X, yc, ya = generate_dataset(a.samples, a.seed)
    order = np.random.default_rng(a.seed+1).permutation(len(X))
    cut = int(.8*len(X))
    tr, va = order[:cut], order[cut:]
    resumed = bool(a.resume and Path(a.resume).exists())
    model = WiFiFixBrain.load(a.resume) if resumed else WiFiFixBrain(a.seed)
    c0, a0 = model.probabilities(X[va])
    before = {"cause_accuracy": accuracy(c0.argmax(1), yc[va]),
              "action_accuracy": accuracy(a0.argmax(1), ya[va])}
    losses = model.fit(X[tr], yc[tr], ya[tr], epochs=a.epochs, seed=a.seed+2)
    pc, pa = model.probabilities(X[va])
    out = Path(a.out)
    model.save(out)
    metrics = {
        "model": "WiFiFixBrain-v1",
        "weights_origin": "own-checkpoint-resume" if resumed else "random-initialization",
        "pretrained_weights_used": False,
        "feature_count": N_FEATURES,
        "cause_classes": list(CAUSES),
        "action_classes": list(ACTIONS),
        "seed": a.seed, "synthetic_samples": a.samples, "train_samples": len(tr),
        "validation_samples": len(va), "epochs": a.epochs,
        "initial_synthetic_validation": before,
        "final_synthetic_validation": {
            "cause_accuracy": accuracy(pc.argmax(1), yc[va]),
            "action_accuracy": accuracy(pa.argmax(1), ya[va]),
            "cause_cross_entropy": float(-np.log(np.maximum(pc[np.arange(len(va)), yc[va]], 1e-8)).mean()),
            "action_cross_entropy": float(-np.log(np.maximum(pa[np.arange(len(va)), ya[va]], 1e-8)).mean()),
        },
        "loss_start": losses[0], "loss_end": losses[-1], "loss_improved": losses[-1] < losses[0],
        "model_bytes": out.stat().st_size, "elapsed_seconds": round(time.time()-started, 2),
        "validation_scope": "synthetic simulator only; not real-world Wi-Fi validation",
        "real_field_examples": 0, "automatic_network_control": "disabled; recommendations are advisory",
    }
    Path(a.metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(a.metrics).write_text(json.dumps(metrics, indent=2)+"\n")
    manifest = {"format": "compressed NumPy NPZ", "model_file": str(out),
                "model_bytes": out.stat().st_size, "model_size_mb": round(out.stat().st_size/1e6, 5),
                "under_18_mb": out.stat().st_size < 18000000,
                "trained_from_scratch": not resumed, "pretrained_model_loaded": False,
                "features": list(FEATURE_NAMES), "scope": metrics["validation_scope"]}
    Path("artifacts/brain_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(metrics, indent=2))
    if not metrics["loss_improved"]:
        raise SystemExit("training loss did not improve")
    if metrics["final_synthetic_validation"]["cause_accuracy"] < .55 or metrics["final_synthetic_validation"]["action_accuracy"] < .45:
        raise SystemExit("synthetic bootstrap validation fell below acceptance thresholds")
if __name__ == "__main__":
    main()
