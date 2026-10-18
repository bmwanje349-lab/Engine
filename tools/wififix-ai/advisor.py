#!/usr/bin/env python3
"""Inference-only adviser. It does not change Wi-Fi settings."""
import argparse
import json
import socket
import sys
from wififix_ai.schema import CAUSES, ACTIONS, encode_observation
from wififix_ai.network import WiFiFixBrain

def gate(obs, cause, cause_conf, action, action_conf):
    if float(obs.get("connected", 1) or 0) < .5:
        return "GATHER_MORE_EVIDENCE", "Wi-Fi is disconnected; no radio toggle will be recommended."
    reliability = float(obs.get("measurement_reliability", 0) or 0)
    if cause_conf < .55 or action_conf < .50 or reliability < .35:
        return "GATHER_MORE_EVIDENCE", "Confidence or measurement coverage is too low."
    rssi = float(obs.get("rssi_dbm", -65) or -65)
    gw_loss = float(obs.get("gateway_loss_pct", -1) if obs.get("gateway_loss_pct") is not None else -1)
    dns = float(obs.get("dns_ok", -1) if obs.get("dns_ok") is not None else -1)
    reconnect_rate = float(obs.get("reconnect_success_rate", -1) if obs.get("reconnect_success_rate") is not None else -1)
    if action == "RECONNECT_WIFI" and (cause != "ROUTER_PATH" or gw_loss < 20 or rssi < -78 or reconnect_rate < .30):
        return "GATHER_MORE_EVIDENCE", "No independent router-path evidence or useful reconnect history."
    if action == "CHECK_DNS" and dns != 0:
        return "GATHER_MORE_EVIDENCE", "DNS failure was not measured."
    if action == "CHECK_SIGNAL" and rssi > -72:
        return "GATHER_MORE_EVIDENCE", "Signal measurement does not support weak-signal diagnosis."
    if action == "WAIT_OR_CONTACT_ISP" and gw_loss >= 20:
        return "GATHER_MORE_EVIDENCE", "Router-path fault is not isolated from an upstream fault."
    return action, "Passed the conservative advisory gate."

def advise(model_path, obs):
    model = WiFiFixBrain.load(model_path)
    cause_probs, action_probs = model.probabilities(encode_observation(obs).reshape(1, -1))
    ci, ai = int(cause_probs[0].argmax()), int(action_probs[0].argmax())
    cause, action = CAUSES[ci], ACTIONS[ai]
    cc, ac = float(cause_probs[0, ci]), float(action_probs[0, ai])
    safe_action, reason = gate(obs, cause, cc, action, ac)
    return {"model": "WiFiFixBrain-v1", "cause": cause, "cause_confidence": round(cc, 4),
            "raw_action": action, "raw_action_confidence": round(ac, 4),
            "advisory_action": safe_action, "gate_reason": reason,
            "automatic_control": False, "field_validated": False,
            "note": "Bootstrap brain trained on synthetic conditions; verify on real Wi-Fi first."}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/brain.npz")
    parser.add_argument("--input-json", default="")
    args = parser.parse_args()
    if args.input_json:
        with open(args.input_json, encoding="utf-8") as stream:
            obs = json.load(stream)
    elif not sys.stdin.isatty():
        raw = sys.stdin.read().strip()
        obs = json.loads(raw) if raw else {}
    else:
        obs = {}
    if "dns_ok" not in obs:
        try:
            socket.getaddrinfo("connectivitycheck.gstatic.com", 443, type=socket.SOCK_STREAM)
            obs["dns_ok"] = 1
            obs.setdefault("measurement_reliability", .55)
        except OSError:
            obs["dns_ok"] = 0
            obs.setdefault("dns_failures", 1)
            obs.setdefault("measurement_reliability", .55)
    print(json.dumps(advise(args.model, obs), sort_keys=True))
if __name__ == "__main__":
    main()
