"""Action-conditioned features for cross-domain value estimation.

The policy scores each observed (state, action) pair. It does not allocate a
fixed neural output neuron to UP/RIGHT/DOWN/LEFT/REST; labels remain environment
controls while this encoder describes what each legal action means in context.
"""
from __future__ import annotations

import math

from .gridworld import ACTIONS, DELTAS

ACTION_FEATURES = 8


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return max(low, min(high, value))


def encode_action_features(features, action) -> tuple[float, ...]:
    """Encode action semantics from a benchmark's fixed 32-value state vector."""
    if len(features) != 32:
        raise ValueError("state features must contain exactly 32 values")
    values = [float(v) for v in features]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("state features must be finite")
    if action not in ACTIONS:
        raise ValueError(f"unknown action label: {action!r}")
    domain = values[16:20]
    if abs(sum(domain) - 1.0) > 1e-5:
        raise ValueError("state features do not identify exactly one supported domain")
    domain_id = max(range(4), key=lambda i: domain[i])
    index = ACTIONS.index(action)

    if domain_id == 0:  # Resource allocation: which project and its marginal utility.
        if action == "REST":
            out = (0.0, 0.0, 0.0, 0.0, 0.0, values[20], values[21], 1.0)
        else:
            out = (
                index / 1.5 - 1.0, values[index], values[8 + index],
                values[4 + index] * 2.0 - 1.0, values[12 + index],
                values[20], values[21], 0.0,
            )
    elif domain_id == 1:  # Scheduling: job identity, duration, priority, slack and readiness.
        if action == "REST":
            out = (0.0, 1.0 / 3.0, 0.0, 0.0, 0.0, 0.0, values[20], 1.0)
        else:
            out = (
                index / 1.5 - 1.0, values[12 + index], values[4 + index],
                values[8 + index], values[24 + index], values[28 + index],
                values[20], 0.0,
            )
    elif domain_id == 2:  # Route planning: geometric displacement, terrain cost and hazard.
        if action == "REST":
            out = (0.0, 0.0, 0.0, 0.0, 0.0, values[22], values[23], 1.0)
        else:
            dx, dy = DELTAS[action]
            out = (
                dx, -dy, values[4 + index], values[8 + index],
                values[28 + index], values[22], values[23], 0.0,
            )
    else:  # Adversarial take-away: remove k stones and encode the resulting position.
        stones = max(0, min(100, int(round(values[20] * 100.0))))
        take = index + 1
        remaining = max(0, stones - take)
        out = (
            take / 5.0, remaining / 100.0, (remaining % 6) / 5.0,
            float(remaining == 0), stones / 100.0, (remaining % 5) / 4.0,
            float(take == 5), float(stones % 5 == 0),
        )
    return tuple(_clip(v) for v in out)
