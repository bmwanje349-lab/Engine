"""Stable, finite feature schema for the WiFiFix neural adviser."""
import math
import numpy as np

CAUSES = (
    "HEALTHY", "WEAK_SIGNAL", "ROUTER_PATH", "DNS_FAILURE",
    "UPSTREAM_FAILURE", "CONGESTION", "INTERNET_OUTAGE", "UNKNOWN",
)
ACTIONS = (
    "NO_ACTION", "RETEST", "CHECK_SIGNAL", "CHECK_DNS",
    "RECONNECT_WIFI", "WAIT_OR_CONTACT_ISP", "GATHER_MORE_EVIDENCE",
)
FEATURE_NAMES = (
    "connected", "rssi_norm", "link_rate_norm", "band_norm",
    "gateway_rtt_norm", "gateway_loss_norm", "internet_rtt_norm",
    "internet_loss_norm", "dns_ok", "download_norm", "baseline_norm",
    "speed_ratio", "recent_disconnects", "reconnect_success_rate",
    "sample_count", "gateway_rtt_trend", "internet_rtt_trend",
    "loss_trend", "rssi_trend", "battery_norm", "router_history_known",
    "gateway_reachable", "internet_reachable", "dns_failures",
    "test_server_ok", "low_power", "rssi_variance", "rtt_variance",
    "drop_rate", "last_reconnect_success", "seconds_since_fix",
    "measurement_reliability",
)
N_FEATURES = len(FEATURE_NAMES)

def number(obs, key, default=None):
    try:
        value = obs.get(key, default)
        if value is None or value == "":
            return default
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default

def clip(value, low, high):
    return max(low, min(high, value))

def log_norm(value, maximum):
    return clip(math.log1p(max(0.0, value)) / math.log1p(maximum), 0.0, 1.0)

def encode_observation(obs):
    """Turn a raw measurement dictionary into 32 bounded float32 features."""
    connected_value = number(obs, "connected")
    # Missing connection state is unknown, not evidence of an active link.
    connected = 0.5 if connected_value is None else clip(connected_value, 0.0, 1.0)
    rssi = number(obs, "rssi_dbm", -65.0)
    rssi = clip(-65.0 if rssi is None else rssi, -100.0, -30.0)
    link = number(obs, "link_speed_mbps", 0.0) or 0.0
    frequency = number(obs, "frequency_mhz", 5000.0)
    frequency = 5000.0 if frequency is None else frequency
    band = 0.0 if frequency < 3000 else (0.5 if frequency < 5925 else 1.0)

    gw_rtt = number(obs, "gateway_rtt_ms")
    gw_loss = number(obs, "gateway_loss_pct")
    net_rtt = number(obs, "internet_rtt_ms")
    net_loss = number(obs, "internet_loss_pct")
    gw_rtt_known = gw_rtt is not None and gw_rtt >= 0
    gw_loss_known = gw_loss is not None and gw_loss >= 0
    net_rtt_known = net_rtt is not None and net_rtt >= 0
    net_loss_known = net_loss is not None and net_loss >= 0
    gw_rtt = clip(gw_rtt, 0, 1000) if gw_rtt_known else 200.0
    gw_loss = clip(gw_loss, 0, 100) if gw_loss_known else 50.0
    net_rtt = clip(net_rtt, 0, 2000) if net_rtt_known else 400.0
    net_loss = clip(net_loss, 0, 100) if net_loss_known else 50.0
    dns = number(obs, "dns_ok", 0.5)
    dns = 0.5 if dns is None or dns < 0 else clip(dns, 0, 1)
    download = max(0.0, number(obs, "download_mbps", 0.0) or 0.0)
    baseline = max(0.0, number(obs, "baseline_mbps", 0.0) or 0.0)
    reconnect_rate = number(obs, "reconnect_success_rate", 0.5)
    reconnect_rate = 0.5 if reconnect_rate is None or reconnect_rate < 0 else clip(reconnect_rate, 0, 1)
    battery = number(obs, "battery_pct", 50.0)
    battery = 50.0 if battery is None else clip(battery, 0, 100)
    server_ok = number(obs, "test_server_ok", 0.5)
    server_ok = 0.5 if server_ok is None or server_ok < 0 else clip(server_ok, 0, 1)
    reliability = number(obs, "measurement_reliability")
    if reliability is None:
        reliability = (int(gw_rtt_known) + int(gw_loss_known) + int(net_rtt_known)
                       + int(net_loss_known) + int(number(obs, "rssi_dbm") is not None)
                       + int(number(obs, "download_mbps") is not None)) / 6.0

    values = [
        connected, (rssi + 100.0) / 70.0, log_norm(link, 1200), band,
        log_norm(gw_rtt, 1000), gw_loss / 100.0, log_norm(net_rtt, 2000),
        net_loss / 100.0, dns, log_norm(download, 1000), log_norm(baseline, 1000),
        clip(download / max(0.5, baseline) if baseline else 1.0, 0, 3) / 3.0,
        clip(number(obs, "recent_disconnects", 0.0) or 0.0, 0, 10) / 10.0,
        reconnect_rate, clip(number(obs, "sample_count", 0.0) or 0.0, 0, 100) / 100.0,
        clip(number(obs, "gateway_rtt_trend_ms", 0.0) or 0.0, -500, 500) / 1000.0 + 0.5,
        clip(number(obs, "internet_rtt_trend_ms", 0.0) or 0.0, -1000, 1000) / 2000.0 + 0.5,
        clip(number(obs, "loss_trend_pct", 0.0) or 0.0, -100, 100) / 200.0 + 0.5,
        clip(number(obs, "rssi_trend_db", 0.0) or 0.0, -30, 30) / 60.0 + 0.5,
        battery / 100.0, 1.0 if number(obs, "router_history_known", 0.0) else 0.0,
        1.0 - gw_loss / 100.0 if gw_loss_known else 0.5,
        1.0 - net_loss / 100.0 if net_loss_known else 0.5,
        clip(number(obs, "dns_failures", 0.0) or 0.0, 0, 5) / 5.0,
        server_ok, clip(number(obs, "low_power", 0.0) or 0.0, 0, 1),
        clip(number(obs, "rssi_variance_db", 5.0) or 0.0, 0, 30) / 30.0,
        clip(number(obs, "rtt_variance_ms", 25.0) or 0.0, 0, 500) / 500.0,
        clip(number(obs, "drop_rate", 0.0) or 0.0, 0, 1),
        clip(number(obs, "last_reconnect_success", 0.5), 0, 1),
        clip(number(obs, "seconds_since_fix", 300.0) or 0.0, 0, 3600) / 3600.0,
        clip(reliability, 0, 1),
    ]
    x = np.asarray(values, dtype=np.float32)
    if x.shape != (N_FEATURES,) or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError("feature encoder produced invalid values")
    return x
