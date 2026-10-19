"""Compact Q-value neural approximator with pure-Python inference.

Training uses NumPy on CI runners. The exported JSON weights are evaluated with
only the Python standard library, so the trained evaluator can run on Termux
without installing NumPy. The network distills the RL table; it is not a
pretrained general-reasoning model.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any

from .gridworld import ACTIONS, GridWorld, State
from .learning import state_features

FORMAT = "strategic-ai-neural-q-v1"
INPUTS, HIDDEN1, HIDDEN2, OUTPUTS = 11, 64, 32, len(ACTIONS)


def _normalise(features: tuple[int, ...] | list[int]) -> list[float]:
    if len(features) != INPUTS:
        raise ValueError(f"expected {INPUTS} state features")
    return [max(-1.0, min(1.0, float(v) / 5.0)) for v in features[:4]] + [
        float(features[4]), float(features[5]) / 5.0, float(features[6]) / 5.0,
        *[float(v) for v in features[7:11]],
    ]


class NeuralPolicy:
    def __init__(self, W1, b1, W2, b2, W3, b3):
        self.W1, self.b1 = W1, b1
        self.W2, self.b2 = W2, b2
        self.W3, self.b3 = W3, b3
        self._cache: dict[tuple[int, ...], list[float]] = {}
        if (len(W1) != INPUTS or len(b1) != HIDDEN1 or len(W2) != HIDDEN1
                or len(b2) != HIDDEN2 or len(W3) != HIDDEN2 or len(b3) != OUTPUTS):
            raise ValueError("neural checkpoint has incorrect layer dimensions")
        if any(len(row) != HIDDEN1 for row in W1) or any(len(row) != HIDDEN2 for row in W2) or any(len(row) != OUTPUTS for row in W3):
            raise ValueError("neural checkpoint contains a malformed matrix")
        for matrix in (W1, W2, W3):
            for row in matrix:
                if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in row):
                    raise ValueError("neural checkpoint contains non-finite weights")
        for vector in (b1, b2, b3):
            if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector):
                raise ValueError("neural checkpoint contains non-finite biases")

    def predict_features(self, features: tuple[int, ...] | list[int]) -> list[float]:
        cache_key = tuple(int(v) for v in features)
        if cache_key in self._cache:
            return list(self._cache[cache_key])
        x = _normalise(features)
        h1 = [max(0.0, sum(x[i] * self.W1[i][j] for i in range(INPUTS)) + self.b1[j])
              for j in range(HIDDEN1)]
        h2 = [max(0.0, sum(h1[i] * self.W2[i][j] for i in range(HIDDEN1)) + self.b2[j])
              for j in range(HIDDEN2)]
        out = [sum(h2[i] * self.W3[i][j] for i in range(HIDDEN2)) + self.b3[j]
               for j in range(OUTPUTS)]
        self._cache[cache_key] = out
        if len(self._cache) > 12000:
            self._cache.clear()
            self._cache[cache_key] = out
        return list(out)

    def prior(self, world: GridWorld, state: State, action: str) -> float:
        if action not in ACTIONS:
            return 0.0
        values = self.predict_features(state_features(world, state))
        legal = [values[ACTIONS.index(a)] for a in world.actions(state)]
        if not legal:
            return 0.0
        center = sum(legal) / len(legal)
        return max(-2.0, min(2.0, (values[ACTIONS.index(action)] - center) / 4.0))

    def save(self, path: str | Path) -> None:
        payload = {"format": FORMAT, "inputs": INPUTS, "hidden": [HIDDEN1, HIDDEN2],
                   "actions": list(ACTIONS), "W1": self.W1, "b1": self.b1,
                   "W2": self.W2, "b2": self.b2, "W3": self.W3, "b3": self.b3,
                   "inference": "pure-python-standard-library"}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "NeuralPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != FORMAT or payload.get("inputs") != INPUTS or payload.get("hidden") != [HIDDEN1, HIDDEN2] or payload.get("actions") != list(ACTIONS):
            raise ValueError("unsupported or incompatible Strategic AI neural checkpoint")
        return cls(payload["W1"], payload["b1"], payload["W2"], payload["b2"], payload["W3"], payload["b3"])

    @classmethod
    def distill(cls, q_table: dict[str, dict[str, float]], *, epochs: int = 45,
                seed: int = 731) -> tuple["NeuralPolicy", dict[str, Any]]:
        """Fit a compact neural approximator to Q-learning values, with held-out loss."""
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("NumPy is required to train the neural evaluator; inference does not require it") from exc
        if epochs < 1 or epochs > 300:
            raise ValueError("epochs must be between 1 and 300")
        rows_x, rows_y, rows_m = [], [], []
        for key, action_values in q_table.items():
            try:
                features = tuple(int(v) for v in key.split(","))
                x = _normalise(features)
            except (TypeError, ValueError):
                continue
            if not action_values:
                continue
            y, mask = [0.0] * OUTPUTS, [0.0] * OUTPUTS
            for action, value in action_values.items():
                if action not in ACTIONS or not isinstance(value, (int, float)) or not math.isfinite(value):
                    continue
                index = ACTIONS.index(action)
                y[index] = max(-30.0, min(30.0, float(value)))
                mask[index] = 1.0
            if sum(mask):
                rows_x.append(x); rows_y.append(y); rows_m.append(mask)
        if len(rows_x) < 20:
            raise ValueError("need at least 20 finite learned Q states to distill a neural evaluator")
        X, Y, M = np.asarray(rows_x, dtype=np.float32), np.asarray(rows_y, dtype=np.float32), np.asarray(rows_m, dtype=np.float32)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(X))
        split = max(1, min(len(order) - 1, int(0.82 * len(order))))
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
        first = {k: np.zeros_like(v) for k, v in params.items()}
        second = {k: np.zeros_like(v) for k, v in params.items()}
        step, batch_size, learning_rate = 0, 96, 0.0015

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
        for _epoch in range(epochs):
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
                    grads = {k: g * (10 / norm) for k, g in grads.items()}
                step += 1
                for name in params:
                    first[name] = 0.9 * first[name] + 0.1 * grads[name]
                    second[name] = 0.999 * second[name] + 0.001 * (grads[name] * grads[name])
                    mhat = first[name] / (1 - 0.9 ** step)
                    vhat = second[name] / (1 - 0.999 ** step)
                    params[name] -= learning_rate * mhat / (np.sqrt(vhat) + 1e-8)
        final_mse = loss(Xv, Yv, Mv)
        if not all(np.isfinite(a).all() for a in params.values()):
            raise ValueError("neural training produced non-finite parameters")
        net = cls(params["W1"].tolist(), params["b1"].tolist(), params["W2"].tolist(),
                  params["b2"].tolist(), params["W3"].tolist(), params["b3"].tolist())
        return net, {"train_states": int(len(train_ix)), "validation_states": int(len(val_ix)),
                     "input_features": INPUTS, "hidden_sizes": [HIDDEN1, HIDDEN2],
                     "parameters": int(sum(v.size for v in params.values())),
                     "initial_validation_mse": round(initial_mse, 6),
                     "final_validation_mse": round(final_mse, 6),
                     "validation_mse_improved": bool(final_mse < initial_mse),
                     "epochs": epochs}
