#!/usr/bin/env python3
"""Inference-only WiFiFix adviser with multi-path diagnostic look-ahead."""
import argparse
import json
import socket
import sys

from wififix_ai.schema import CAUSES, ACTIONS, encode_observation, number
from wififix_ai.network import WiFiFixBrain, MODEL_NAME
from wififix_ai.planner import plan_diagnostics


def gate(obs, cause, cause_conf, action, action_conf):
    connected = number(obs, "connected")
    if connected is None:
        return "GATHER_MORE_EVIDENCE", "Connection state was not measured; no diagnosis should assume Wi-Fi is connected."
    if connected < .5:
        return "GATHER_MORE_EVIDENCE", "Wi-Fi is disconnected; no radio toggle will be recommended."

    reliability_value = number(obs, "measurement_reliability", 0.0)
    reliability = 0.0 if reliability_value is None else reliability_value
    if cause_conf < .55 or action_conf < .50 or reliability < .35:
        return "GATHER_MORE_EVIDENCE", "Confidence or measurement coverage is too low."

    rssi_value = number(obs, "rssi_dbm")
    rssi = -101.0 if rssi_value is None else rssi_value
    gw_loss_value = number(obs, "gateway_loss_pct")
    gw_loss = -1.0 if gw_loss_value is None else gw_loss_value
    dns_value = number(obs, "dns_ok")
    dns = -1.0 if dns_value is None else dns_value
    reconnect_value = number(obs, "reconnect_success_rate")
    reconnect_rate = -1.0 if reconnect_value is None else reconnect_value

    if action == "RECONNECT_WIFI" and (
        cause != "ROUTER_PATH" or gw_loss < 20 or rssi < -78 or reconnect_rate < .30
    ):
        return "GATHER_MORE_EVIDENCE", "No independent router-path evidence or useful reconnect history."
    if action == "CHECK_DNS" and dns != 0:
        return "GATHER_MORE_EVIDENCE", "DNS failure was not measured."
    if action == "CHECK_SIGNAL" and (rssi_value is None or rssi > -72):
        return "GATHER_MORE_EVIDENCE", "Signal measurement does not support weak-signal diagnosis."
    if action == "WAIT_OR_CONTACT_ISP" and gw_loss >= 20:
        return "GATHER_MORE_EVIDENCE", "Router-path fault is not isolated from an upstream fault."
    return action, "Passed the conservative advisory gate."


def maybe_probe_dns(obs, enabled=False):
    """Optional DNS probe; never infer DNS failure unless an IP path first works."""
    if not enabled:
        return "not_requested"
    if not isinstance(obs, dict) or not obs:
        return "skipped_no_observation"

    connected = number(obs, "connected")
    if connected is None or connected < .5:
        return "skipped_connection_unconfirmed"
    existing_dns = number(obs, "dns_ok")
    if existing_dns in (0.0, 1.0):
        return "skipped_dns_already_measured"

    # A failed hostname lookup alone cannot distinguish DNS failure from no internet.
    # First require a successful connection to a literal IP address.
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=2):
            pass
    except OSError:
        return "skipped_ip_connectivity_unconfirmed"

    try:
        socket.getaddrinfo("connectivitycheck.gstatic.com", 443, type=socket.SOCK_STREAM)
    except OSError:
        obs["dns_ok"] = 0
        previous_failures = number(obs, "dns_failures", 0.0)
        obs["dns_failures"] = max(1.0, 0.0 if previous_failures is None else previous_failures)
        obs.setdefault("measurement_reliability", .55)
        return "failed_after_ip_connectivity_confirmed"

    obs["dns_ok"] = 1
    obs.setdefault("measurement_reliability", .55)
    return "resolved_after_ip_connectivity_confirmed"

def advise(model_path, obs):
    model = WiFiFixBrain.load(model_path)
    cause_probs, action_probs = model.probabilities(encode_observation(obs).reshape(1, -1))
    ci, ai = int(cause_probs[0].argmax()), int(action_probs[0].argmax())
    cause, action = CAUSES[ci], ACTIONS[ai]
    cc, ac = float(cause_probs[0, ci]), float(action_probs[0, ai])
    safe_action, reason = gate(obs, cause, cc, action, ac)
    planning = plan_diagnostics(cause_probs[0], observation=obs, max_depth=3, top_paths=3)
    preferred = planning["candidate_paths"][0] if planning["candidate_paths"] else None
    return {
        "model": MODEL_NAME,
        "cause": cause,
        "cause_confidence": round(cc, 4),
        "raw_action": action,
        "raw_action_confidence": round(ac, 4),
        "advisory_action": safe_action,
        "gate_reason": reason,
        "recommended_next_test": preferred["test"] if preferred else None,
        "candidate_diagnostic_paths": planning["candidate_paths"],
        "diagnostic_search": {
            "strategy": planning["strategy"],
            "search_depth": planning["search_depth"],
            "root_entropy_bits": planning["root_entropy_bits"],
            "top_cause_hypotheses": planning["cause_prior"][:3],
            "interpretation": planning["interpretation"],
        },
        "automatic_control": False,
        "field_validated": False,
        "note": (
            "The neural model is trained on synthetic conditions unless verified field examples "
            "were explicitly supplied. Candidate paths are modeled diagnostic options, not guarantees."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/brain.npz")
    parser.add_argument("--input-json", default="")
    parser.add_argument("--probe-dns", action="store_true", help="Opt in to a guarded DNS probe; only a DNS failure after IP connectivity succeeds is recorded.")
    args = parser.parse_args()
    if args.input_json:
        with open(args.input_json, encoding="utf-8") as stream:
            obs = json.load(stream)
    elif not sys.stdin.isatty():
        raw = sys.stdin.read().strip()
        obs = json.loads(raw) if raw else {}
    else:
        obs = {}
    if not isinstance(obs, dict):
        raise SystemExit("input JSON must be an observation object")
    dns_probe_status = maybe_probe_dns(obs, enabled=args.probe_dns)
    result = advise(args.model, obs)
    result["dns_probe_status"] = dns_probe_status
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
