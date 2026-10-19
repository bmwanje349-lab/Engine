"""Action-conditioned cross-domain Q/value network.

Training requires NumPy; JSON inference deliberately uses only the Python
standard library so the same checkpoint can run in Android Termux. The Q head
scores any encoded (state, action) pair and the value head estimates a state
value for PUCT leaf evaluation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .action_encoding import ACTION_FEATURES, encode_action_features
from .gridworld import ACTIONS

INPUTS, HIDDEN1, HIDDEN2 = 32, 96, 64
NETWORK_INPUTS = INPUTS + ACTION_FEATURES
FORMAT = "strategic-ai-general-neural-v4-puct"
def normalized_action_prior(action_values, action, *, limit: float = 2.0) -> float:
    """Convert return-scale Q estimates into bounded, relative PUCT prior logits.

    Raw returns have different scales across benchmark families. Feeding them
    directly into softmax can make a model's prior almost deterministic before
    search has tested the branch. Per-state standardization preserves ordering
    while preventing large rewards from overwhelming tree exploration.
    """
    if action not in action_values or not action_values:
        return 0.0
    scores = [float(value) for value in action_values.values() if math.isfinite(float(value))]
    if not scores:
        return 0.0
    mean = sum(scores) / len(scores)
    variance = sum((value - mean) ** 2 for value in scores) / len(scores)
    scale = math.sqrt(variance)
    if not math.isfinite(scale) or scale < 1e-8:
        return 0.0
    score = (float(action_values[action]) - mean) / scale
    return max(-limit, min(limit, score))

LEGACY_FORMAT = "strategic-ai-general-neural-v3"


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


def _action_features(values):
    if len(values) != ACTION_FEATURES:
        raise ValueError(f"expected {ACTION_FEATURES} state-action features")
    clean = []
    for value in values:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("state-action features must be finite")
        clean.append(max(-1.0, min(1.0, value)))
    return clean


def _split_target_key(key: str):
    # Production data prefixes each state with an independent scenario ID.
    # Older hand-built tests use a feature-only key; each such row is its own
    # group, never pretending that untagged rows share a known scenario.
    if "|" in key:
        group, feature_text = key.split("|", 1)
    else:
        group, feature_text = key, key
    values = _features([float(v) for v in feature_text.split(",")])
    return group, values


class GeneralNeuralPolicy:
    """Compact state/action scorer plus a state-value head (standard-library inference)."""

    def __init__(self, W1, b1, W2, b2, Wq, bq, Wv, bv, trained_domains=None,
                 q_mean=0.0, q_scale=1.0, value_mean=0.0, value_scale=1.0):
        self.trained_domains = frozenset(
            range(4) if trained_domains is None else (int(v) for v in trained_domains)
        )
        if not self.trained_domains or any(value not in range(4) for value in self.trained_domains):
            raise ValueError("general neural checkpoint has invalid trained-domain metadata")
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2
        self.Wq, self.bq, self.Wv, self.bv = Wq, bq, Wv, bv
        self.q_mean, self.q_scale = float(q_mean), float(q_scale)
        self.value_mean, self.value_scale = float(value_mean), float(value_scale)
        if (not all(math.isfinite(v) for v in (self.q_mean, self.q_scale, self.value_mean, self.value_scale))
                or self.q_scale <= 0 or self.value_scale <= 0):
            raise ValueError("general neural checkpoint has invalid target scaling")
        if (len(W1) != NETWORK_INPUTS or len(b1) != HIDDEN1
                or len(W2) != HIDDEN1 or len(b2) != HIDDEN2
                or len(Wq) != HIDDEN2 or len(bq) != 1
                or len(Wv) != HIDDEN2 or len(bv) != 1):
            raise ValueError("general neural checkpoint has incorrect layer dimensions")
        if (any(len(row) != HIDDEN1 for row in W1)
                or any(len(row) != HIDDEN2 for row in W2)):
            raise ValueError("general neural checkpoint contains a malformed matrix")
        for matrix in (W1, W2):
            for row in matrix:
                if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in row):
                    raise ValueError("general neural checkpoint contains non-finite weights")
        for vector in (b1, b2, Wq, Wv, bq, bv):
            if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector):
                raise ValueError("general neural checkpoint contains non-finite weights")
        self._hidden_cache: dict[tuple[float, ...], list[float]] = {}

    def supports_features(self, features) -> bool:
        """Return whether the checkpoint was trained on this state's domain."""
        values = _features(features)
        domain = values[16:20]
        if abs(sum(domain) - 1.0) > 1e-5 or max(domain) < 0.5:
            return False
        domain_id = max(range(4), key=lambda index: domain[index])
        return domain_id in self.trained_domains

    def _hidden(self, features, action_features):
        state = _features(features)
        action = _action_features(action_features)
        x = state + action
        cache_key = tuple(x)
        cached = self._hidden_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        h1 = [
            max(0.0, sum(x[i] * self.W1[i][j] for i in range(NETWORK_INPUTS)) + self.b1[j])
            for j in range(HIDDEN1)
        ]
        h2 = [
            max(0.0, sum(h1[i] * self.W2[i][j] for i in range(HIDDEN1)) + self.b2[j])
            for j in range(HIDDEN2)
        ]
        if len(self._hidden_cache) >= 16000:
            self._hidden_cache.clear()
        self._hidden_cache[cache_key] = h2
        return list(h2)

    def predict_action_value(self, features, action_features, *, action_name=None) -> float:
        """Estimate Q(s,a) in the target's original return units."""
        if isinstance(action_features, str):
            action_name = action_features
            action_features = encode_action_features(features, action_features)
        h2 = self._hidden(features, action_features)
        standardized = sum(h2[i] * self.Wq[i] for i in range(HIDDEN2)) + self.bq[0]
        result = self.q_mean + self.q_scale * standardized
        if not math.isfinite(result):
            return 0.0
        return float(result)

    def predict_value_features(self, features) -> float:
        """Estimate V(s) in return units; zero action features form the value-head input."""
        h2 = self._hidden(features, (0.0,) * ACTION_FEATURES)
        standardized = sum(h2[i] * self.Wv[i] for i in range(HIDDEN2)) + self.bv[0]
        result = self.value_mean + self.value_scale * standardized
        if not math.isfinite(result):
            return 0.0
        return float(result)

    def predict_features(self, features) -> list[float]:
        """Compatibility helper; production decisions use explicit state-action scoring."""
        return [
            self.predict_action_value(features, encode_action_features(features, action), action_name=action)
            for action in ACTIONS
        ]

    def save(self, path: str | Path) -> None:
        payload = {
            "format": FORMAT,
            "inputs": INPUTS,
            "action_features": ACTION_FEATURES,
            "hidden": [HIDDEN1, HIDDEN2],
            "W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
            "Wq": self.Wq, "bq": self.bq, "Wv": self.Wv, "bv": self.bv,
            "trained_domains": sorted(self.trained_domains),
            "target_scaling": {
                "q_mean": self.q_mean, "q_scale": self.q_scale,
                "value_mean": self.value_mean, "value_scale": self.value_scale,
            },
            "target_semantics": {
                "policy": "action-conditioned Q(s,a) in normalized-return units",
                "value": "state value V(s) for PUCT leaf evaluation",
            },
            "inference": "pure-python-standard-library",
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") == LEGACY_FORMAT:
            return LegacyGeneralNeuralPolicy(payload)
        if (payload.get("format") != FORMAT or payload.get("inputs") != INPUTS
                or payload.get("action_features") != ACTION_FEATURES
                or payload.get("hidden") != [HIDDEN1, HIDDEN2]):
            raise ValueError("unsupported or incompatible general Strategic AI checkpoint")
        domains = payload.get("trained_domains")
        if not isinstance(domains, list):
            raise ValueError("general neural checkpoint is missing trained-domain metadata")
        scaling = payload.get("target_scaling") or {}
        return cls(
            payload["W1"], payload["b1"], payload["W2"], payload["b2"],
            payload["Wq"], payload["bq"], payload["Wv"], payload["bv"],
            trained_domains=domains, q_mean=scaling.get("q_mean", 0.0),
            q_scale=scaling.get("q_scale", 1.0), value_mean=scaling.get("value_mean", 0.0),
            value_scale=scaling.get("value_scale", 1.0),
        )

    @classmethod
    def distill(cls, targets: dict[str, dict[str, float]], *, epochs: int = 50,
                seed: int = 731) -> tuple["GeneralNeuralPolicy", dict[str, Any]]:
        """Fit Q(s,a) and V(s) using a scenario-disjoint train/validation split."""
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("NumPy is required to train the general evaluator; inference does not require it") from exc
        if not 1 <= epochs <= 300:
            raise ValueError("epochs must be between 1 and 300")

        parsed = []
        for key, action_values in targets.items():
            try:
                group, features = _split_target_key(key)
            except (TypeError, ValueError):
                continue
            values = {}
            for action, raw_value in action_values.items():
                if action not in ACTIONS:
                    continue
                if isinstance(raw_value, dict):
                    raw_value = raw_value.get("value")
                if not isinstance(raw_value, (int, float)) or not math.isfinite(float(raw_value)):
                    continue
                values[action] = float(raw_value)
            if values:
                parsed.append({"key": key, "group": group, "features": features, "values": values})
        if len(parsed) < 40:
            raise ValueError(f"need at least 40 search-labeled states, found {len(parsed)}")

        groups = sorted({row["group"] for row in parsed})
        if len(groups) < 2:
            raise ValueError("scenario-disjoint validation requires at least two independent scenario groups")
        rng = np.random.default_rng(seed)
        group_order = list(rng.permutation(groups))
        validation_group_count = max(1, min(len(group_order) - 1, int(round(0.16 * len(group_order)))))
        validation_groups = set(group_order[-validation_group_count:])
        train_groups = set(group_order[:-validation_group_count])
        train_rows = [row for row in parsed if row["group"] in train_groups]
        val_rows = [row for row in parsed if row["group"] in validation_groups]
        if not train_rows or not val_rows:
            raise ValueError("scenario split produced an empty training or validation set")

        q_train_raw = [value for row in train_rows for value in row["values"].values()]
        v_train_raw = [max(row["values"].values()) for row in train_rows]
        q_mean = float(np.mean(q_train_raw))
        q_scale = float(np.std(q_train_raw))
        v_mean = float(np.mean(v_train_raw))
        v_scale = float(np.std(v_train_raw))
        q_scale = q_scale if math.isfinite(q_scale) and q_scale > 1e-6 else 1.0
        v_scale = v_scale if math.isfinite(v_scale) and v_scale > 1e-6 else 1.0

        def make_q_arrays(rows):
            xs, ys = [], []
            for row in rows:
                for action, value in row["values"].items():
                    action_vector = encode_action_features(row["features"], action)
                    xs.append(row["features"] + list(action_vector))
                    ys.append([(value - q_mean) / q_scale])
            return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

        def make_v_arrays(rows):
            xs, ys = [], []
            zero_action = [0.0] * ACTION_FEATURES
            for row in rows:
                xs.append(row["features"] + zero_action)
                ys.append([(max(row["values"].values()) - v_mean) / v_scale])
            return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

        Xq, Yq = make_q_arrays(train_rows)
        Xqv, Yqv = make_q_arrays(val_rows)
        Xv, Yv = make_v_arrays(train_rows)
        Xvv, Yvv = make_v_arrays(val_rows)
        params = {
            "W1": rng.normal(0, math.sqrt(2 / NETWORK_INPUTS), (NETWORK_INPUTS, HIDDEN1)).astype(np.float32),
            "b1": np.zeros(HIDDEN1, np.float32),
            "W2": rng.normal(0, math.sqrt(2 / HIDDEN1), (HIDDEN1, HIDDEN2)).astype(np.float32),
            "b2": np.zeros(HIDDEN2, np.float32),
            "Wq": rng.normal(0, math.sqrt(1 / HIDDEN2), (HIDDEN2, 1)).astype(np.float32),
            "bq": np.zeros(1, np.float32),
            "Wv": rng.normal(0, math.sqrt(1 / HIDDEN2), (HIDDEN2, 1)).astype(np.float32),
            "bv": np.zeros(1, np.float32),
        }
        first = {name: np.zeros_like(value) for name, value in params.items()}
        second = {name: np.zeros_like(value) for name, value in params.items()}
        step, batch_size, learning_rate = 0, 128, 0.0012

        def forward(x):
            z1 = x @ params["W1"] + params["b1"]
            h1 = np.maximum(z1, 0)
            z2 = h1 @ params["W2"] + params["b2"]
            h2 = np.maximum(z2, 0)
            return h2, (x, z1, h1, z2)

        def loss(x, y, head):
            h2, _ = forward(x)
            prediction = h2 @ params["W" + head] + params["b" + head]
            return float(np.mean((prediction - y) ** 2))

        def train_batch(x, y, head):
            nonlocal step
            h2, (x0, z1, h1, z2) = forward(x)
            out_weight, out_bias = "W" + head, "b" + head
            prediction = h2 @ params[out_weight] + params[out_bias]
            dout = (2.0 * (prediction - y)) / max(1, len(y))
            grads = {
                out_weight: h2.T @ dout,
                out_bias: dout.sum(axis=0),
            }
            dz2 = (dout @ params[out_weight].T) * (z2 > 0)
            grads["W2"], grads["b2"] = h1.T @ dz2, dz2.sum(axis=0)
            dz1 = (dz2 @ params["W2"].T) * (z1 > 0)
            grads["W1"], grads["b1"] = x0.T @ dz1, dz1.sum(axis=0)
            norm = math.sqrt(sum(float((grad * grad).sum()) for grad in grads.values()))
            if norm > 10.0:
                grads = {name: grad * (10.0 / norm) for name, grad in grads.items()}
            step += 1
            for name, grad in grads.items():
                first[name] = 0.9 * first[name] + 0.1 * grad
                second[name] = 0.999 * second[name] + 0.001 * (grad * grad)
                mhat = first[name] / (1.0 - 0.9 ** step)
                vhat = second[name] / (1.0 - 0.999 ** step)
                params[name] -= learning_rate * mhat / (np.sqrt(vhat) + 1e-8)

        initial_q_loss = loss(Xqv, Yqv, "q")
        initial_v_loss = loss(Xvv, Yvv, "v")
        initial_loss = 0.5 * (initial_q_loss + initial_v_loss)
        best_loss = initial_loss
        best_q_loss, best_v_loss = initial_q_loss, initial_v_loss
        best = {name: value.copy() for name, value in params.items()}
        best_epoch = 0
        for epoch in range(epochs):
            for x, y, head in ((Xq, Yq, "q"), (Xv, Yv, "v")):
                order = rng.permutation(len(x))
                for start in range(0, len(order), batch_size):
                    indices = order[start:start + batch_size]
                    train_batch(x[indices], y[indices], head)
            q_loss, v_loss = loss(Xqv, Yqv, "q"), loss(Xvv, Yvv, "v")
            candidate_loss = 0.5 * (q_loss + v_loss)
            if math.isfinite(candidate_loss) and candidate_loss < best_loss:
                best_loss, best_q_loss, best_v_loss = candidate_loss, q_loss, v_loss
                best_epoch = epoch + 1
                best = {name: value.copy() for name, value in params.items()}
        params = best
        if not all(np.isfinite(array).all() for array in params.values()):
            raise ValueError("general neural training produced non-finite parameters")
        trained_domains = sorted({
            max(range(4), key=lambda index: float(row["features"][16 + index]))
            for row in train_rows
        })
        net = cls(
            params["W1"].tolist(), params["b1"].tolist(),
            params["W2"].tolist(), params["b2"].tolist(),
            params["Wq"].reshape(-1).tolist(), params["bq"].reshape(-1).tolist(),
            params["Wv"].reshape(-1).tolist(), params["bv"].reshape(-1).tolist(),
            trained_domains=trained_domains, q_mean=q_mean, q_scale=q_scale,
            value_mean=v_mean, value_scale=v_scale,
        )
        return net, {
            "train_states": len(train_rows), "validation_states": len(val_rows),
            "train_scenarios": len(train_groups), "validation_scenarios": len(validation_groups),
            "scenario_group_overlap": bool(train_groups & validation_groups),
            "input_features": INPUTS, "action_features": ACTION_FEATURES,
            "hidden_sizes": [HIDDEN1, HIDDEN2],
            "parameters": int(sum(value.size for value in params.values())),
            "initial_validation_mse": round(initial_loss, 6),
            "final_validation_mse": round(best_loss, 6),
            "initial_q_validation_mse": round(initial_q_loss, 6),
            "final_q_validation_mse": round(best_q_loss, 6),
            "initial_value_validation_mse": round(initial_v_loss, 6),
            "final_value_validation_mse": round(best_v_loss, 6),
            "validation_mse_improved": bool(best_loss < initial_loss),
            "best_epoch": best_epoch, "epochs": epochs,
            "target_scaling": "global train-scenario standardization; inverse-scaled at inference",
            "target_semantics": "action-conditioned Q(s,a) plus state-value V(s)",
            "trained_domains": trained_domains,
        }


class LegacyGeneralNeuralPolicy:
    """Read-only adapter so a v3 release can be the first v4 champion baseline."""

    def __init__(self, payload):
        self.trained_domains = frozenset(int(v) for v in payload.get("trained_domains", range(4)))
        self.W1, self.b1 = payload["W1"], payload["b1"]
        self.W2, self.b2 = payload["W2"], payload["b2"]
        self.W3, self.b3 = payload["W3"], payload["b3"]
        self._cache = {}

    def supports_features(self, features):
        values = _features(features)
        domain = values[16:20]
        return (abs(sum(domain) - 1.0) <= 1e-5
                and max(range(4), key=lambda i: domain[i]) in self.trained_domains)

    def predict_features(self, features):
        x = _features(features)
        key = tuple(x)
        if key in self._cache:
            return list(self._cache[key])
        h1 = [max(0.0, sum(x[i] * self.W1[i][j] for i in range(INPUTS)) + self.b1[j])
              for j in range(96)]
        h2 = [max(0.0, sum(h1[i] * self.W2[i][j] for i in range(96)) + self.b2[j])
              for j in range(64)]
        out = [sum(h2[i] * self.W3[i][j] for i in range(64)) + self.b3[j] for j in range(5)]
        self._cache[key] = out
        if len(self._cache) > 16000:
            self._cache.clear()
        return list(out)

    def predict_action_value(self, features, action_features, *, action_name=None):
        action = action_name or (action_features if isinstance(action_features, str) else None)
        if action not in ACTIONS:
            action = ACTIONS[0]
        return float(self.predict_features(features)[ACTIONS.index(action)])

    def predict_value_features(self, features):
        return max(self.predict_features(features))

