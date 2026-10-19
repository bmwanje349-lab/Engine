"""Domain-aware semantic features for action-conditioned value estimation.

The network scores an arbitrary named state/action pair. Its output is not a
fixed five-action grid head: each benchmark exposes its own legal action
vocabulary, and this encoder maps those actions to relevant contextual features.
"""
from __future__ import annotations

import math

from .gridworld import ACTIONS, DELTAS

ACTION_FEATURES = 8
RESOURCE_ACTIONS = (
    "ALLOCATE_PROJECT_1", "ALLOCATE_PROJECT_2",
    "ALLOCATE_PROJECT_3", "ALLOCATE_PROJECT_4", "SKIP_ALLOCATION",
)
SCHEDULING_ACTIONS = (
    "SCHEDULE_JOB_1", "SCHEDULE_JOB_2",
    "SCHEDULE_JOB_3", "SCHEDULE_JOB_4", "WAIT",
)
ROUTE_ACTIONS = ACTIONS
NIM_ACTIONS = ("TAKE_1", "TAKE_2", "TAKE_3", "TAKE_4", "TAKE_5")
ACTIONS_BY_DOMAIN = {
    0: RESOURCE_ACTIONS,
    1: SCHEDULING_ACTIONS,
    2: ROUTE_ACTIONS,
    3: NIM_ACTIONS,
}


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return max(low, min(high, value))


def actions_for_features(features) -> tuple[str, ...]:
    """Return the domain-specific action vocabulary identified by state features."""
    if len(features) != 32:
        raise ValueError("state features must contain exactly 32 values")
    values = [float(v) for v in features]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("state features must be finite")
    domain = values[16:20]
    if abs(sum(domain) - 1.0) > 1e-5 or max(domain) < 0.5:
        raise ValueError("state features do not identify exactly one supported domain")
    return ACTIONS_BY_DOMAIN[max(range(4), key=lambda i: domain[i])]


def legacy_action_index(features, action: str) -> int:
    """Map semantic domain actions to the equivalent index in a legacy v3 head."""
    actions = actions_for_features(features)
    if action not in actions:
        raise ValueError(f"action {action!r} does not belong to this state domain")
    return actions.index(action)


def encode_action_features(features, action) -> tuple[float, ...]:
    """Encode the meaning of one legal action in the context of a 32-value state."""
    values = [float(v) for v in features]
    actions = actions_for_features(values)
    if action not in actions:
        raise ValueError(f"unknown action {action!r} for this state domain")
    domain = values[16:20]
    domain_id = max(range(4), key=lambda i: domain[i])
    index = actions.index(action)

    if domain_id == 0:  # Choose a project or stop allocating.
        if action == "SKIP_ALLOCATION":
            out = (0.0, 0.0, 0.0, 0.0, 0.0, values[20], values[21], 1.0)
        else:
            out = (
                index / 1.5 - 1.0, values[index], values[8 + index],
                values[4 + index] * 2.0 - 1.0, values[12 + index],
                values[20], values[21], 0.0,
            )
    elif domain_id == 1:  # Schedule a particular job or wait one unit.
        if action == "WAIT":
            out = (0.0, 1.0 / 3.0, 0.0, 0.0, 0.0, 0.0, values[20], 1.0)
        else:
            out = (
                index / 1.5 - 1.0, values[12 + index], values[4 + index],
                values[8 + index], values[24 + index], values[28 + index],
                values[20], 0.0,
            )
    elif domain_id == 2:  # Route directions or an energy-restoring action.
        if action == "REST":
            out = (0.0, 0.0, 0.0, 0.0, 0.0, values[22], values[23], 1.0)
        else:
            dx, dy = DELTAS[action]
            out = (
                dx, -dy, values[4 + index], values[8 + index],
                values[28 + index], values[22], values[23], 0.0,
            )
    else:  # Adversarial take-away: explicitly encode the chosen removal size.
        stones = max(0, min(100, int(round(values[20] * 100.0))))
        take = index + 1
        remaining = max(0, stones - take)
        out = (
            take / 5.0, remaining / 100.0, (remaining % 6) / 5.0,
            float(remaining == 0), stones / 100.0, (remaining % 5) / 4.0,
            float(take == 5), float(stones % 5 == 0),
        )
    return tuple(_clip(v) for v in out)
