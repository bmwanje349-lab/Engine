"""Cross-domain action-value network with domain-aware features and ranking-scaled targets.

Uses NumPy only for training. Saved JSON checkpoints use standard-library inference
so they can be loaded by the Android Termux CLI without NumPy.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

INPUTS, HIDDEN1, HIDDEN2, OUTPUTS = 32, 96, 64, 5
FORMAT = "strategic-ai-general-neural-v3"


def _features(values):
    if len(values) != INPUTS:
        raise ValueError(f"expected {INPUTS} general-state features")
    clean = []
    for value in values:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("general-state features must be finite")
        clean.append(max(-1.0, min(1.0, value)))
    return clean


class GeneralNeuralPolicy:
    """Small 32-96-64-5 MLP; output values are state-relative action advantages."""

    def __init__(self, W1, b1, W2, b2, W3, b3, trained_domains=None):
        self.trained_domains = frozenset(range(4) if trained_domains is None else (int(v) for v in trained_domains))
        if not self.trained_domains or any(value not in range(4) for value in self.trained_domains):
            raise ValueError("general neural checkpoint has invalid trained-domain metadata")
        self.W1, self.b1 = W1, b1
        self.W2, self.b2 = W2, b2
        self.W3, self.b3 = W3, b3
        self._cache: dict[tuple[float, ...], list[float]] = {}
        if (len(W1) != INPUTS or len(b1) != HIDDEN1
                or len(W2) != HIDDEN1 or len(b2) != HIDDEN2
                or len(W3) != HIDDEN2 or len(b3) != OUTPUTS):
            raise ValueError("general neural checkpoint has incorrect layer dimensions")
        if any(len(row) != HIDDEN1 for row in W1) or any(len(row) != HIDDEN2 for row in W2) or any(len(row) != OUTPUTS for row in W3):
            raise ValueError("general neural checkpoint contains a malformed matrix")
        for matrix in (W1, W2, W3):
            for row in matrix:
                if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in row):
                    raise ValueError("general neural checkpoint contains non-finite weights")
        for vector in (b1, b2, b3):
            if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector):
                raise ValueError("general neural checkpoint contains non-finite biases")

    def supports_features(self, features) -> bool:
        """Return whether this checkpoint was trained on the feature's domain."""
        values = _features(features)
        domain = values[16:20]
        if len(domain) != 4 or abs(sum(domain) - 1.0) > 1e-5 or max(domain) < 0.5:
            return False
        domain_id = max(range(4), key=lambda index: domain[index])
        return domain_id in self.trained_domains

    def predict_features(self, features) -> list[float]:
        x = _features(features)
        cache_key = tuple(x)
        if cache_key in self._cache:
            return list(self._cache[cache_key])
        h1 = [max(0.0, sum(x[i] * self.W1[i][j] for i in range(INPUTS)) + self.b1[j])
              for j in range(HIDDEN1)]
        h2 = [max(0.0, sum(h1[i] * self.W2[i][j] for i in range(HIDDEN1)) + self.b2[j])
              for j in range(HIDDEN2)]
        out = [sum(h2[i] * self.W3[i][j] for i in range(HIDDEN2)) + self.b3[j]
               for j in range(OUTPUTS)]
        self._cache[cache_key] = out
        if len(self._cache) > 16000:
            self._cache.clear()
            self._cache[cache_key] = out
        return list(out)

    def save(self, path: str | Path) -> None:
        payload = {
            "format": FORMAT, "inputs": INPUTS,
            "hidden": [HIDDEN1, HIDDEN2], "outputs": OUTPUTS,
            "W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
            "W3": self.W3, "b3": self.b3,
            "trained_domains": sorted(self.trained_domains),
            "target_semantics": "per-state standardized action-value advantage",
            "inference": "pure-python-standard-library",
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "GeneralNeuralPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if (payload.get("format") != FORMAT or payload.get("inputs") != INPUTS
                or payload.get("hidden") != [HIDDEN1, HIDDEN2]
                or payload.get("outputs") != OUTPUTS):
            raise ValueError("unsupported or incompatible general Strategic AI checkpoint")
        domains = payload.get("trained_domains")
        if not isinstance(domains, list):
            raise ValueError("general neural checkpoint is missing trained-domain metadata")
        return cls(payload["W1"], payload["b1"], payload["W2"], payload["b2"],
                   payload["W3"], payload["b3"], trained_domains=domains)

    @classmethod
    def distill(cls, targets: dict[str, dict[str, float]], *, epochs: int = 50,
                seed: int = 731) -> tuple["GeneralNeuralPolicy", dict[str, Any]]:
        """Fit state-relative action preferences so rewards of different scales balance."""
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("NumPy is required to train the general evaluator; inference does not require it") from exc
        if not 1 <= epochs <= 300:
            raise ValueError("epochs must be between 1 and 300")

        actions = ("UP", "RIGHT", "DOWN", "LEFT", "REST")
        rows_x, rows_y, rows_m = [], [], []
        for key, action_values in targets.items():
            try:
                features = _features([float(v) for v in key.split(",")])
            except (TypeError, ValueError):
                continue
            y, mask = [0.0] * OUTPUTS, [0.0] * OUTPUTS
            for action, value in action_values.items():
                if action not in actions or not isinstance(value, (int, float)) or not math.isfinite(value):
                    continue
                index = actions.index(action)
                y[index], mask[index] = float(value), 1.0
            indices = [i for i, present in enumerate(mask) if present]
            if not indices:
                continue
            # Standardize only valid actions for this state. This teaches which
            # action is better without allowing large-reward domains to dominate.
            mean = sum(y[i] for i in indices) / len(indices)
            variance = sum((y[i] - mean) ** 2 for i in indices) / len(indices)
            scale = math.sqrt(variance)
            for i in indices:
                y[i] = (max(-4.0, min(4.0, (y[i] - mean) / scale))
                        if scale > 1e-8 else 0.0)
            rows_x.append(features)
            rows_y.append(y)
            rows_m.append(mask)
        if len(rows_x) < 40:
            raise ValueError(f"need at least 40 search-labeled states, found {len(rows_x)}")

        X = np.asarray(rows_x, dtype=np.float32)
        Y = np.asarray(rows_y, dtype=np.float32)
        M = np.asarray(rows_m, dtype=np.float32)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(X))
        split = max(1, min(len(order) - 1, int(0.84 * len(order))))
        train_ix, val_ix = order[:split], order[split:]
        Xtr, Ytr, Mtr = X[train_ix], Y[train_ix], M[train_ix]
        Xv, Yv, Mv = X[val_ix], Y[val_ix], M[val_ix]
        params = {
            "W1": rng.normal(0, math.sqrt(2 / INPUTS), (INPUTS, HIDDEN1)).astype(np.float32),
            "b1": np.zeros(HIDDEN1, np.float32),
            "W2": rng.normal(0, math.sqrt(2 / HIDDEN1), (HIDDEN1, HIDDEN2)).astype(np.float32),
            "b2": np.zeros(HIDDEN2, np.float32),
            "W3": rng.normal(0, math.sqrt(1 / HIDDEN2), (HIDDEN2, OUTPUTS)).astype(np.float32),
            "b3": np.zeros(OUTPUTS, np.float32),
        }
        first = {name: np.zeros_like(value) for name, value in params.items()}
        second = {name: np.zeros_like(value) for name, value in params.items()}
        step, batch_size, learning_rate = 0, 128, 0.0012

        def forward(x):
            z1 = x @ params["W1"] + params["b1"]
            h1 = np.maximum(z1, 0)
            z2 = h1 @ params["W2"] + params["b2"]
            h2 = np.maximum(z2, 0)
            out = h2 @ params["W3"] + params["b3"]
            return out, (x, z1, h1, z2, h2)

        def loss(x, y, mask):
            pred, _ = forward(x)
            denom = max(1.0, float(mask.sum()))
            return float((((pred - y) * mask) ** 2).sum() / denom)

        initial_mse = loss(Xv, Yv, Mv)
        best_loss = initial_mse
        best = {name: value.copy() for name, value in params.items()}
        best_epoch = 0
        for epoch in range(epochs):
            perm = rng.permutation(len(Xtr))
            for start in range(0, len(perm), batch_size):
                ix = perm[start:start + batch_size]
                x, y, mask = Xtr[ix], Ytr[ix], Mtr[ix]
                out, (_, z1, h1, z2, h2) = forward(x)
                denom = max(1.0, float(mask.sum()))
                dout = (2.0 * (out - y) * mask) / denom
                grads = {}
                grads["W3"], grads["b3"] = h2.T @ dout, dout.sum(axis=0)
                dz2 = (dout @ params["W3"].T) * (z2 > 0)
                grads["W2"], grads["b2"] = h1.T @ dz2, dz2.sum(axis=0)
                dz1 = (dz2 @ params["W2"].T) * (z1 > 0)
                grads["W1"], grads["b1"] = x.T @ dz1, dz1.sum(axis=0)
                norm = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
                if norm > 10:
                    grads = {name: grad * (10 / norm) for name, grad in grads.items()}
                step += 1
                for name in params:
                    first[name] = 0.9 * first[name] + 0.1 * grads[name]
                    second[name] = 0.999 * second[name] + 0.001 * (grads[name] * grads[name])
                    mhat = first[name] / (1 - 0.9 ** step)
                    vhat = second[name] / (1 - 0.999 ** step)
                    params[name] -= learning_rate * mhat / (np.sqrt(vhat) + 1e-8)
            candidate_loss = loss(Xv, Yv, Mv)
            if math.isfinite(candidate_loss) and candidate_loss < best_loss:
                best_loss = candidate_loss
                best_epoch = epoch + 1
                best = {name: value.copy() for name, value in params.items()}
        params = best
        if not all(np.isfinite(array).all() for array in params.values()):
            raise ValueError("general neural training produced non-finite parameters")
        trained_domains = sorted({
            max(range(4), key=lambda index: float(features[16 + index]))
            for features in rows_x
        })
        net = cls(params["W1"].tolist(), params["b1"].tolist(), params["W2"].tolist(),
                  params["b2"].tolist(), params["W3"].tolist(), params["b3"].tolist(),
                  trained_domains=trained_domains)
        return net, {
            "train_states": int(len(train_ix)), "validation_states": int(len(val_ix)),
            "input_features": INPUTS, "hidden_sizes": [HIDDEN1, HIDDEN2],
            "parameters": int(sum(value.size for value in params.values())),
            "initial_validation_mse": round(initial_mse, 6),
            "final_validation_mse": round(best_loss, 6),
            "validation_mse_improved": bool(best_loss < initial_mse),
            "best_epoch": best_epoch, "epochs": epochs,
            "target_scaling": "per-state standardized action-value advantage",
            "trained_domains": trained_domains,
        }
