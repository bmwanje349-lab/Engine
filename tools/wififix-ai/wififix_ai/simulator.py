"""Causal synthetic Wi-Fi simulator. Its labels are not evidence from real routers."""
import numpy as np
from .schema import CAUSES, ACTIONS, encode_observation

def u(rng, low, high):
    return float(rng.uniform(low, high))

def generate_one(rng, cause):
    obs = {
        "connected": 1, "rssi_dbm": u(rng, -68, -42),
        "link_speed_mbps": u(rng, 36, 300),
        "frequency_mhz": float(rng.choice([2412, 2437, 5180, 5200, 5745, 5975])),
        "gateway_rtt_ms": u(rng, 2, 18), "gateway_loss_pct": u(rng, 0, 2),
        "internet_rtt_ms": u(rng, 15, 90), "internet_loss_pct": u(rng, 0, 3),
        "dns_ok": 1, "download_mbps": u(rng, 12, 120), "baseline_mbps": u(rng, 15, 130),
        "recent_disconnects": int(rng.integers(0, 2)), "reconnect_success_rate": u(rng, .25, .95),
        "sample_count": int(rng.integers(3, 70)), "gateway_rtt_trend_ms": u(rng, -5, 5),
        "internet_rtt_trend_ms": u(rng, -15, 15), "loss_trend_pct": u(rng, -2, 2),
        "rssi_trend_db": u(rng, -3, 3), "battery_pct": u(rng, 10, 100),
        "router_history_known": int(rng.integers(0, 2)), "dns_failures": 0, "test_server_ok": 1,
        "low_power": int(rng.random() < .1), "rssi_variance_db": u(rng, 1, 9),
        "rtt_variance_ms": u(rng, 3, 60), "drop_rate": u(rng, 0, .05),
        "last_reconnect_success": u(rng, .4, 1), "seconds_since_fix": u(rng, 120, 3600),
        "measurement_reliability": u(rng, .75, 1),
    }
    if cause == "HEALTHY":
        obs["download_mbps"] = max(3, obs["baseline_mbps"] * u(rng, .65, 1.25))
    elif cause == "WEAK_SIGNAL":
        obs.update(rssi_dbm=u(rng, -94, -74), link_speed_mbps=u(rng, 1, 36),
                   gateway_rtt_ms=u(rng, 5, 80), gateway_loss_pct=u(rng, 0, 18),
                   internet_rtt_ms=u(rng, 30, 240), internet_loss_pct=u(rng, 0, 15),
                   download_mbps=u(rng, .1, 12), rssi_variance_db=u(rng, 5, 22),
                   rssi_trend_db=u(rng, -12, 1), recent_disconnects=int(rng.integers(1, 9)),
                   drop_rate=u(rng, .1, .8))
    elif cause == "ROUTER_PATH":
        obs.update(rssi_dbm=u(rng, -70, -40), gateway_rtt_ms=u(rng, 45, 500),
                   gateway_loss_pct=u(rng, 20, 100), internet_rtt_ms=u(rng, 100, 900),
                   internet_loss_pct=u(rng, 20, 100), download_mbps=u(rng, 0, 8),
                   recent_disconnects=int(rng.integers(0, 6)), drop_rate=u(rng, .05, .9),
                   rtt_variance_ms=u(rng, 80, 400), router_history_known=1,
                   reconnect_success_rate=u(rng, .05, .95), last_reconnect_success=u(rng, 0, 1))
    elif cause == "DNS_FAILURE":
        obs.update(rssi_dbm=u(rng, -70, -40), gateway_rtt_ms=u(rng, 1, 18),
                   gateway_loss_pct=u(rng, 0, 2), internet_rtt_ms=u(rng, 15, 90),
                   internet_loss_pct=u(rng, 0, 4), dns_ok=0, dns_failures=int(rng.integers(1, 6)),
                   download_mbps=u(rng, 5, 100), test_server_ok=0)
    elif cause == "UPSTREAM_FAILURE":
        obs.update(rssi_dbm=u(rng, -70, -40), gateway_rtt_ms=u(rng, 1, 20),
                   gateway_loss_pct=u(rng, 0, 4), internet_rtt_ms=u(rng, 300, 1800),
                   internet_loss_pct=u(rng, 60, 100), download_mbps=u(rng, 0, 2),
                   test_server_ok=0, dns_ok=int(rng.random() > .4), dns_failures=int(rng.integers(0, 3)))
    elif cause == "CONGESTION":
        obs.update(rssi_dbm=u(rng, -70, -40), gateway_rtt_ms=u(rng, 20, 150),
                   gateway_loss_pct=u(rng, 0, 10), internet_rtt_ms=u(rng, 130, 550),
                   internet_loss_pct=u(rng, 0, 12), download_mbps=u(rng, .1, 20),
                   baseline_mbps=u(rng, 30, 180), rtt_variance_ms=u(rng, 50, 350),
                   internet_rtt_trend_ms=u(rng, 20, 500), drop_rate=u(rng, 0, .25))
    elif cause == "INTERNET_OUTAGE":
        obs.update(rssi_dbm=u(rng, -70, -40), gateway_rtt_ms=u(rng, 1, 18),
                   gateway_loss_pct=u(rng, 0, 2), internet_rtt_ms=u(rng, 900, 2000),
                   internet_loss_pct=u(rng, 85, 100), download_mbps=0,
                   test_server_ok=0, dns_ok=int(rng.random() > .5))
    elif cause == "UNKNOWN":
        for key in ("gateway_rtt_ms", "gateway_loss_pct", "internet_rtt_ms",
                    "internet_loss_pct", "download_mbps"):
            if rng.random() < .55:
                obs.pop(key, None)
        obs.update(dns_ok=-1, measurement_reliability=u(rng, .05, .45),
                   recent_disconnects=int(rng.integers(0, 10)), reconnect_success_rate=-1)
        if rng.random() < .4:
            obs["connected"] = 0
    action = {
        "HEALTHY": "NO_ACTION", "WEAK_SIGNAL": "CHECK_SIGNAL", "ROUTER_PATH": "RECONNECT_WIFI",
        "DNS_FAILURE": "CHECK_DNS", "UPSTREAM_FAILURE": "WAIT_OR_CONTACT_ISP",
        "CONGESTION": "RETEST", "INTERNET_OUTAGE": "WAIT_OR_CONTACT_ISP",
        "UNKNOWN": "GATHER_MORE_EVIDENCE",
    }[cause]
    if cause == "ROUTER_PATH" and (obs["reconnect_success_rate"] < .3 or obs["rssi_dbm"] < -78):
        action = "GATHER_MORE_EVIDENCE"
    return obs, CAUSES.index(cause), ACTIONS.index(action)

def generate_dataset(count, seed=1234):
    rng = np.random.default_rng(seed)
    weights = np.asarray([.16, .15, .14, .12, .14, .12, .09, .08], dtype=float)
    weights /= weights.sum()
    X = np.empty((count, 32), dtype=np.float32)
    cause_y = np.empty(count, dtype=np.int64)
    action_y = np.empty(count, dtype=np.int64)
    for i in range(count):
        obs, cause_y[i], action_y[i] = generate_one(rng, CAUSES[int(rng.choice(len(CAUSES), p=weights))])
        X[i] = encode_observation(obs)
    return X, cause_y, action_y
