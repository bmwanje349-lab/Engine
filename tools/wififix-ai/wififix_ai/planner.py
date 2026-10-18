"""Bounded expectimax-style diagnostic search over alternative Wi-Fi test paths.

This is a planning layer, not a claim that simulated outcomes are guaranteed.
It uses the neural model's cause probabilities as a prior and transparent,
reviewable likelihood tables to reason over alternative test results.
"""
import math
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
                 ("sample_count", "gateway_rtt_trend_ms", "internet_rtt_trend_ms", "loss_trend_pct")),
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


def _measurement_value(observation, key):
    """Return a finite, valid numeric measurement, or None for missing sentinels."""
    if not isinstance(observation, dict):
        return None
    raw = observation.get(key)
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    if key == "dns_ok" and value not in (0.0, 1.0):
        return None
    if key in {
        "gateway_rtt_ms", "gateway_loss_pct", "internet_rtt_ms",
        "internet_loss_pct", "download_mbps", "baseline_mbps", "sample_count",
    } and value < 0:
        return None
    return value


def _observed_test_result(test, observation):
    """Infer a conservative result for tests already represented in an observation.

    None means there is not enough measured information. 'inconclusive' means a
    complete measurement exists but does not cross a decision threshold.
    """
    if not isinstance(observation, dict) or not observation:
        return None
    key = test["key"]

    if key == "CHECK_SIGNAL":
        rssi = _measurement_value(observation, "rssi_dbm")
        if rssi is None or not -100 <= rssi <= -20:
            return None
        return "weak_signal" if rssi <= -72 else "signal_adequate"

    if key == "PING_GATEWAY":
        rtt = _measurement_value(observation, "gateway_rtt_ms")
        loss = _measurement_value(observation, "gateway_loss_pct")
        if loss is not None and loss >= 20:
            return "gateway_slow_or_lossy"
        if rtt is not None and rtt >= 100:
            return "gateway_slow_or_lossy"
        if rtt is None or loss is None:
            return None
        if loss <= 5 and rtt <= 30:
            return "gateway_healthy"
        return "inconclusive"

    if key == "PING_INTERNET":
        rtt = _measurement_value(observation, "internet_rtt_ms")
        loss = _measurement_value(observation, "internet_loss_pct")
        if loss is not None and loss >= 50:
            return "internet_slow_or_unreachable"
        if rtt is not None and rtt >= 500:
            return "internet_slow_or_unreachable"
        if rtt is None or loss is None:
            return None
        if loss <= 5 and rtt <= 150:
            return "internet_reachable"
        return "inconclusive"

    if key == "CHECK_DNS":
        dns = _measurement_value(observation, "dns_ok")
        if dns is None:
            return None
        return "dns_works" if dns == 1 else "dns_fails"

    if key == "SPEED_TEST":
        download = _measurement_value(observation, "download_mbps")
        baseline = _measurement_value(observation, "baseline_mbps")
        if download is None or baseline is None or baseline <= 0:
            return None
        ratio = download / baseline
        if ratio < .5:
            return "speed_far_below_baseline"
        if ratio >= .7:
            return "speed_near_baseline"
        return "inconclusive"

    if key == "RETEST_STABILITY":
        count = _measurement_value(observation, "sample_count")
        gateway_trend = _measurement_value(observation, "gateway_rtt_trend_ms")
        internet_trend = _measurement_value(observation, "internet_rtt_trend_ms")
        loss_trend = _measurement_value(observation, "loss_trend_pct")
        if (count is None or count < 3 or gateway_trend is None
                or internet_trend is None or loss_trend is None):
            return None
        if abs(gateway_trend) >= 50 or abs(internet_trend) >= 100 or abs(loss_trend) >= 10:
            return "unstable"
        if abs(gateway_trend) <= 20 and abs(internet_trend) <= 40 and abs(loss_trend) <= 5:
            return "stable"
        return "inconclusive"

    return None


def _observed_test_penalty(test, observation):
    """Strongly de-prioritise a partly measured test instead of a tiny repeat fee."""
    known = sum(_measurement_value(observation, key) is not None
                for key in test["measurement_keys"])
    return .35 if known else 0.0


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
    obs = observation if isinstance(observation, dict) else {}
    observed_tests = []
    partial_tests = []
    remaining_tests = []
    for test in TESTS:
        outcome = _observed_test_result(test, obs)
        if outcome is not None:
            observed_tests.append({
                "test": test["key"],
                "outcome": outcome,
                "measurement_keys": list(test["measurement_keys"]),
            })
        else:
            remaining_tests.append(test)
            if _observed_test_penalty(test, obs) > 0:
                partial_tests.append({
                    "test": test["key"],
                    "available_measurements": [
                        key for key in test["measurement_keys"]
                        if _measurement_value(obs, key) is not None
                    ],
                })

    roots = []
    for test in remaining_tests:
        remaining = tuple(item for item in remaining_tests if item["key"] != test["key"])
        roots.append(_evaluate_test(belief, test, remaining, int(max_depth), obs))
    roots.sort(key=lambda item: item["search_score"], reverse=True)
    return {
        "strategy": "observation_aware_bounded_expectimax",
        "search_depth": int(max_depth),
        "root_entropy_bits": round(entropy_bits(belief), 4),
        "cause_prior": _top_causes(belief, limit=len(CAUSES)),
        "observed_tests": observed_tests,
        "partial_tests": partial_tests,
        "remaining_tests": [test["key"] for test in remaining_tests],
        "candidate_paths": [
            {"rank": rank, **node} for rank, node in enumerate(roots[:int(top_paths)], start=1)
        ],
        "evidence_handling": (
            "The neural distribution is treated as already conditioned on the observation "
            "features supplied to it. Completed tests are reported and excluded instead of "
            "Bayes-updating those same readings a second time; only hypothetical outcomes "
            "of remaining tests update beliefs inside search branches."
        ),
        "interpretation": (
            "Branches show modeled outcome probabilities and updated cause hypotheses. "
            "They are decision-support estimates, not guaranteed real-world outcomes."
        ),
    }
