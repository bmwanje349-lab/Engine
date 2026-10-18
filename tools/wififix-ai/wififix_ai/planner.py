"""Bounded expectimax-style diagnostic search over alternative Wi-Fi test paths.

This is a planning layer, not a claim that simulated outcomes are guaranteed.
It uses the neural model's cause probabilities as a prior and transparent,
reviewable likelihood tables to reason over alternative test results.
"""
import numpy as np
from .schema import CAUSES


# Class order matches schema.CAUSES:
# HEALTHY, WEAK_SIGNAL, ROUTER_PATH, DNS_FAILURE, UPSTREAM_FAILURE,
# CONGESTION, INTERNET_OUTAGE, UNKNOWN.
def _binary_test(key, title, positive, negative, positive_likelihood, cost, measurement_keys):
    p = np.asarray(positive_likelihood, dtype=np.float64)
    if p.shape != (len(CAUSES),) or np.any(p <= 0) or np.any(p >= 1):
        raise ValueError("invalid test likelihood table")
    return {
        "key": key,
        "title": title,
        "positive": positive,
        "negative": negative,
        "likelihoods": {positive: p, negative: 1.0 - p},
        "cost": float(cost),
        "measurement_keys": tuple(measurement_keys),
    }


TESTS = (
    _binary_test("CHECK_SIGNAL", "Recheck Wi-Fi signal", "weak_signal", "signal_adequate",
                 [.04, .88, .12, .04, .04, .10, .04, .15], .15, ("rssi_dbm",)),
    _binary_test("PING_GATEWAY", "Test the local router path", "gateway_slow_or_lossy", "gateway_healthy",
                 [.03, .20, .88, .03, .04, .32, .03, .20], .35,
                 ("gateway_rtt_ms", "gateway_loss_pct")),
    _binary_test("PING_INTERNET", "Test external reachability", "internet_slow_or_unreachable", "internet_reachable",
                 [.03, .17, .65, .04, .92, .65, .98, .25], .45,
                 ("internet_rtt_ms", "internet_loss_pct")),
    _binary_test("CHECK_DNS", "Check DNS resolution", "dns_fails", "dns_works",
                 [.02, .03, .04, .94, .40, .03, .35, .18], .20, ("dns_ok",)),
    _binary_test("SPEED_TEST", "Compare speed with baseline", "speed_far_below_baseline", "speed_near_baseline",
                 [.04, .83, .91, .08, .96, .82, .99, .35], .90,
                 ("download_mbps", "baseline_mbps")),
    _binary_test("RETEST_STABILITY", "Repeat tests to detect instability", "unstable", "stable",
                 [.08, .74, .70, .06, .82, .78, .87, .35], .65,
                 ("sample_count", "gateway_rtt_trend_ms", "internet_rtt_trend_ms")),
)


def entropy_bits(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def _normalise(prior):
    p = np.asarray(prior, dtype=np.float64).reshape(-1)
    if p.shape != (len(CAUSES),) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("cause probabilities must be eight finite non-negative values")
    total = float(p.sum())
    if total <= 0:
        p = np.full(len(CAUSES), 1.0 / len(CAUSES))
    else:
        p = p / total
    # Prevent a synthetic-trained model from treating one uncalibrated guess
    # as absolute truth before additional diagnostic evidence is collected.
    p = .85 * p + .15 / len(CAUSES)
    return p / p.sum()


def _top_causes(p, limit=3):
    order = np.argsort(p)[::-1][:limit]
    return [{"cause": CAUSES[int(i)], "probability": round(float(p[i]), 4)} for i in order]


def _observed_test_penalty(test, observation):
    if not observation:
        return 0.0
    keys = test["measurement_keys"]
    known = all(k in observation and observation[k] is not None and observation[k] != "" for k in keys)
    try:
        reliability = float(observation.get("measurement_reliability", 0.0) or 0.0)
    except (TypeError, ValueError):
        reliability = 0.0
    # A repeat is still an option because conditions change. Prefer tests that
    # add missing evidence when both options have similar expected value.
    return .07 if known and reliability >= .75 else 0.0


def _updated_belief(belief, likelihood):
    unnormalised = belief * likelihood
    total = float(unnormalised.sum())
    if total <= 1e-12:
        return belief.copy(), 0.0
    return unnormalised / total, total


def _evaluate_test(belief, test, remaining, depth, observation):
    start_entropy = entropy_bits(belief)
    branch_records = {}
    expected_entropy = 0.0
    expected_future = 0.0

    for outcome, likelihood in test["likelihoods"].items():
        posterior, outcome_probability = _updated_belief(belief, likelihood)
        outcome_probability = max(0.0, min(1.0, outcome_probability))
        child = None
        if depth > 1 and remaining and outcome_probability >= .025:
            child = _best_node(posterior, remaining, depth - 1, observation)
            expected_future += outcome_probability * child["search_score"]
        expected_entropy += outcome_probability * entropy_bits(posterior)
        branch_records[outcome] = {
            "probability": round(float(outcome_probability), 4),
            "most_likely_causes": _top_causes(posterior),
            "next_step": child,
        }

    information_gain = max(0.0, start_entropy - expected_entropy)
    repeated_measurement_penalty = _observed_test_penalty(test, observation)
    score = information_gain + .60 * expected_future - .018 * test["cost"] - repeated_measurement_penalty
    return {
        "test": test["key"],
        "title": test["title"],
        "expected_information_gain_bits": round(float(information_gain), 4),
        "search_score": round(float(score), 4),
        "branches": branch_records,
    }


def _best_node(belief, remaining, depth, observation):
    if depth < 1 or not remaining:
        return None
    candidates = []
    for test in remaining:
        next_remaining = tuple(item for item in remaining if item["key"] != test["key"])
        candidates.append(_evaluate_test(belief, test, next_remaining, depth, observation))
    return max(candidates, key=lambda item: item["search_score"])


def plan_diagnostics(cause_probabilities, observation=None, max_depth=3, top_paths=3):
    """Return several candidate plans, each with contingent outcome branches."""
    if not 1 <= int(max_depth) <= 4:
        raise ValueError("search depth must be between 1 and 4")
    if not 1 <= int(top_paths) <= len(TESTS):
        raise ValueError("top_paths must be between 1 and the number of tests")
    belief = _normalise(cause_probabilities)
    roots = []
    for test in TESTS:
        remaining = tuple(item for item in TESTS if item["key"] != test["key"])
        roots.append(_evaluate_test(belief, test, remaining, int(max_depth), observation))
    roots.sort(key=lambda item: item["search_score"], reverse=True)
    return {
        "strategy": "bounded_expectimax_with_bayesian_branch_updates",
        "search_depth": int(max_depth),
        "root_entropy_bits": round(entropy_bits(belief), 4),
        "cause_prior": _top_causes(belief, limit=len(CAUSES)),
        "candidate_paths": [
            {"rank": rank, **node} for rank, node in enumerate(roots[:int(top_paths)], start=1)
        ],
        "interpretation": (
            "Branches show modeled outcome probabilities and updated cause hypotheses. "
            "They are decision-support estimates, not guaranteed real-world outcomes."
        ),
    }
