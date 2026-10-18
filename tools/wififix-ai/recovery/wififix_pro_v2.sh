#!/data/data/com.termux/files/usr/bin/bash
# wififix_pro_v2.sh - conservative adaptive Wi-Fi monitor/recovery engine for Termux.
# It does not forget saved networks. Automatic Wi-Fi toggling is attempted only
# after repeated, measured failures. Android may block Wi-Fi control without root/ADB.
# Commands: run|start|stop|status|stats|diagnose|pause|resume
# Recommended: pkg install termux-api jq curl iproute2

export LC_ALL=C
umask 077

# Tunables (may be overridden via environment variables)
MIN_MBPS=${MIN_MBPS:-5}
ADAPT_FRAC=${ADAPT_FRAC:-0.4}
WEAK_RSSI=${WEAK_RSSI:-78}
CYCLE=${CYCLE:-15}
HEARTBEAT=${HEARTBEAT:-300}
SUSPECT_GAP=${SUSPECT_GAP:-20}
TEST_BYTES=${TEST_BYTES:-500000}
NEED=${NEED:-3}
BASE_COOL=${BASE_COOL:-90}
MAX_COOL=${MAX_COOL:-900}
SETTLE=${SETTLE:-6}
LIVE_NOTIF=${LIVE_NOTIF:-1}
INET_HOSTS=${INET_HOSTS:-"1.1.1.1 8.8.8.8"}
SPEED_URL=${SPEED_URL:-https://speed.cloudflare.com/__down}
CHECK_URL=${CHECK_URL:-https://connectivitycheck.gstatic.com/generate_204}

D="$HOME/.wififix"
mkdir -p "$D/nets" || exit 1
chmod 700 "$D" "$D/nets" 2>/dev/null || :
LOG="$D/wififix.log"; PIDF="$D/pid"; PAUSE="$D/pause"; LAST="$D/last"
SELF=$(readlink -f "$0" 2>/dev/null || printf '%s' "$0")

declare -A RL=()
METHODS=(); METHOD=none; WIFI_OFF=0; LOWPOWER=0
W_UP=0; W_SSID=""; W_RSSI=""; W_LINK=""; W_FREQ=""; W_IP=""; W_STATE=""
G_LOSS=-1; G_RTT=""; I_LOSS=-1; I_RTT=""; P_LOSS=-1; P_RTT=""; DL=""; thr=$MIN_MBPS
NET_STATUS=-1 # 1=internet verified, 0=internet failure evidenced, -1=inconclusive
N_KEY=""; NF=""; N_BASE=0; N_SAMP=0; N_LINKBEST=0; N_OK=0; N_FAIL=0; N_RTT=0; N_GWOK=0
bad=0; episode_fail=0; cool=$BASE_COOL; COOL_UNTIL=0; STATE=OK; LAST_DL="-"
V_CODE=OK; V_TEXT=""; V_FIX=1; RECON=0; ADB_SERIAL="${ADB_SERIAL:-}"

now()   { date +%s; }
isnum() { [[ "${1:-}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; }
isint() { [[ "${1:-}" =~ ^-?[0-9]+$ ]]; }
flt()   { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a<b)}'; }
fge()   { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a>=b)}'; }
calc()  { awk "BEGIN { printf \"%.2f\", $1 }" 2>/dev/null; }
nap()   { sleep "$1" & wait $! 2>/dev/null; return 0; }

log() {
  if [ -f "$LOG" ] && [ "$(wc -c <"$LOG" 2>/dev/null || echo 0)" -gt 1048576 ]; then
    mv -f "$LOG" "$LOG.1" 2>/dev/null || :
  fi
  printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"
}

due() {
  local t; t=$(now)
  if [ $((t - ${RL[$1]:-0})) -ge "$2" ]; then RL[$1]=$t; return 0; fi
  return 1
}
notify() {
  command -v termux-notification >/dev/null 2>&1 || return 0
  termux-notification --id wififix_event --title "$1" --content "$2" >/dev/null 2>&1 &
}
live() {
  [ "$LIVE_NOTIF" = 1 ] && [ "$LOWPOWER" != 1 ] || return 0
  command -v termux-notification >/dev/null 2>&1 || return 0
  termux-notification --id wififix_live --ongoing --alert-once --priority low \
    --title "$1" --content "$2" >/dev/null 2>&1 &
}
need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing $1. Install with: pkg install $2"; exit 1; }; }
running() {
  [ -f "$PIDF" ] || return 1
  local p; p=$(cat "$PIDF" 2>/dev/null)
  isint "$p" && [ "$p" -gt 1 ] && kill -0 "$p" 2>/dev/null
}

# Validate environment overrides, preventing zero-length loops and bad arithmetic.
validate_config() {
  isnum "$MIN_MBPS" && fge "$MIN_MBPS" 0.1 || MIN_MBPS=5
  isnum "$ADAPT_FRAC" && fge "$ADAPT_FRAC" 0.1 && fge 1 "$ADAPT_FRAC" || ADAPT_FRAC=0.4
  isint "$WEAK_RSSI" && [ "$WEAK_RSSI" -ge 40 ] && [ "$WEAK_RSSI" -le 95 ] || WEAK_RSSI=78
  isint "$CYCLE" && [ "$CYCLE" -ge 5 ] && [ "$CYCLE" -le 120 ] || CYCLE=15
  isint "$HEARTBEAT" && [ "$HEARTBEAT" -ge 60 ] || HEARTBEAT=300
  isint "$SUSPECT_GAP" && [ "$SUSPECT_GAP" -ge 10 ] || SUSPECT_GAP=20
  isint "$TEST_BYTES" && [ "$TEST_BYTES" -ge 50000 ] && [ "$TEST_BYTES" -le 2000000 ] || TEST_BYTES=500000
  isint "$NEED" && [ "$NEED" -ge 2 ] && [ "$NEED" -le 8 ] || NEED=3
  isint "$BASE_COOL" && [ "$BASE_COOL" -ge 30 ] || BASE_COOL=90
  isint "$MAX_COOL" && [ "$MAX_COOL" -ge "$BASE_COOL" ] || MAX_COOL=900
  isint "$SETTLE" && [ "$SETTLE" -ge 1 ] && [ "$SETTLE" -le 30 ] || SETTLE=6
}
validate_config

# ------------------------------ Wi-Fi state
read_wifi() {
  W_UP=0; W_SSID=""; W_RSSI=""; W_LINK=""; W_FREQ=""; W_IP=""; W_STATE=""
  local json out
  json=$(timeout 10 termux-wifi-connectioninfo 2>/dev/null) || return 1
  out=$(printf '%s' "$json" | jq -r '[
      (.rssi // ""),
      (if (.rx_link_speed_mbps // -1) > 0 then .rx_link_speed_mbps else (.link_speed_mbps // "") end),
      (.frequency_mhz // ""), (.ip // ""), (.supplicant_state // ""), (.ssid // "")
    ] | map(tostring) | join("|")' 2>/dev/null) || return 1
  IFS='|' read -r W_RSSI W_LINK W_FREQ W_IP W_STATE W_SSID <<<"$out"
  isint "$W_RSSI" || W_RSSI=""
  isint "$W_LINK" && [ "$W_LINK" -gt 0 ] || W_LINK=""
  isint "$W_FREQ" || W_FREQ=""
  if [ "$W_STATE" = COMPLETED ] && [ -n "$W_IP" ] && [ "$W_IP" != 0.0.0.0 ]; then W_UP=1; fi
  return 0
}
net_key() {
  case "$W_SSID" in ""|"<unknown ssid>") echo unknown ;; *) printf '%s' "$W_SSID" ;; esac
}
same_net() { [ -z "$1" ] || [ "$1" = "<unknown ssid>" ] || [ "$W_SSID" = "$1" ]; }
band() {
  if ! isint "${1:-}"; then echo "?"
  elif [ "$1" -ge 5925 ]; then echo 6GHz
  elif [ "$1" -ge 4900 ]; then echo 5GHz
  else echo 2.4GHz; fi
}

# ------------------------------ per-network learned profile
load_net() {
  N_KEY=$1
  NF="$D/nets/$(printf '%s' "$1" | md5sum | cut -c1-12)"
  N_BASE=0; N_SAMP=0; N_LINKBEST=0; N_OK=0; N_FAIL=0; N_RTT=0; N_GWOK=0
  [ -f "$NF" ] || return 0
  # Never source a state file as shell code; accept only known, bounded values.
  local key value
  while IFS='=' read -r key value; do
    case "$key" in
      N_BASE|N_LINKBEST|N_RTT)
        if isnum "$value" && awk -v n="$value" 'BEGIN{exit !(n>=0 && n<1000000)}'; then
          printf -v "$key" '%s' "$value"
        fi ;;
      N_SAMP|N_OK|N_FAIL)
        if isint "$value" && [ "$value" -ge 0 ] && [ "$value" -le 100000000 ]; then
          printf -v "$key" '%s' "$value"
        fi ;;
      N_GWOK) case "$value" in 0|1) N_GWOK=$value ;; esac ;;
    esac
  done < "$NF"
}
save_net() {
  {
    printf '# SSID: %s\n' "$(printf '%s' "$N_KEY" | tr -d '\r\n')"
    printf 'N_BASE=%s\nN_SAMP=%s\nN_LINKBEST=%s\nN_OK=%s\nN_FAIL=%s\nN_RTT=%s\nN_GWOK=%s\n' \
      "$N_BASE" "$N_SAMP" "$N_LINKBEST" "$N_OK" "$N_FAIL" "$N_RTT" "$N_GWOK"
  } >"$NF.tmp" && chmod 600 "$NF.tmp" && mv -f "$NF.tmp" "$NF"
}
learn() {
  local d
  if [ "$N_SAMP" -eq 0 ]; then N_BASE=$1
  elif fge "$1" "$N_BASE"; then N_BASE=$(calc "0.5*$N_BASE+0.5*$1")
  else N_BASE=$(calc "0.9*$N_BASE+0.1*$1"); fi
  N_SAMP=$((N_SAMP + 1))
  if [ -n "$W_LINK" ]; then
    d=$(calc "$N_LINKBEST*0.999")
    if fge "$W_LINK" "$d"; then N_LINKBEST=$W_LINK; else N_LINKBEST=$d; fi
  fi
  if [ -n "$I_RTT" ]; then
    if fge "$N_RTT" 1; then N_RTT=$(calc "0.8*$N_RTT+0.2*$I_RTT"); else N_RTT=$I_RTT; fi
  fi
  save_net
}
threshold() {
  thr=$MIN_MBPS
  if [ "$N_SAMP" -ge 3 ]; then
    local a; a=$(calc "$ADAPT_FRAC*$N_BASE")
    if fge "$a" "$thr"; then thr=$a; fi
  fi
}

# ------------------------------ probes
# probe HOST COUNT -> P_LOSS percent (-1 means ping unavailable), P_RTT ms
probe() {
  local out
  out=$(timeout 20 ping -c "$2" -i 0.3 -W 2 "$1" 2>/dev/null)
  if ! printf '%s' "$out" | grep -qi 'packet loss'; then
    out=$(timeout 20 ping -c "$2" "$1" 2>/dev/null)
  fi
  P_LOSS=$(printf '%s\n' "$out" | awk '/packet loss/ {for(i=1;i<=NF;i++) if($i ~ /%$/){gsub(/%/,"",$i); print $i; exit}}')
  P_RTT=$(printf '%s\n' "$out" | awk '/min\/avg\/max/ {sub(/.*= */ , ""); split($0,a,"/"); print a[2]; exit}')
  P_LOSS=${P_LOSS%.*}
  isint "$P_LOSS" || P_LOSS=-1
  isnum "$P_RTT" || P_RTT=""
}

gw_addr() {
  local g hex iface dest flags
  if command -v ip >/dev/null 2>&1; then
    g=$(ip -4 route show default 2>/dev/null | awk '$1=="default" {for(i=1;i<=NF;i++) if($i=="via"){print $(i+1); exit}}')
    [ -n "$g" ] && { printf '%s' "$g"; return 0; }
  fi
  # Fallback for Android builds where iproute output is unavailable.
  if [ -r /proc/net/route ]; then
    while read -r iface dest hex flags _; do
      [ "$dest" = 00000000 ] && [ -n "$hex" ] && [ "$hex" != 00000000 ] || continue
      printf '%d.%d.%d.%d' "$((16#${hex:6:2}))" "$((16#${hex:4:2}))" "$((16#${hex:2:2}))" "$((16#${hex:0:2}))"
      return 0
    done < <(tail -n +2 /proc/net/route 2>/dev/null)
  fi
  return 1
}
https_check() {
  local code
  code=$(curl -L -sS --connect-timeout 4 --max-time 7 -o /dev/null -w '%{http_code}' "$CHECK_URL" 2>/dev/null) || code=000
  case "$code" in 2??|3??) return 0 ;; *) return 1 ;; esac
}
cheap_probe() {
  local gw h any_ping=0 internet_ping=0
  gw=$(gw_addr 2>/dev/null) || gw=""
  if [ -n "$gw" ]; then
    probe "$gw" 3; G_LOSS=$P_LOSS; G_RTT=$P_RTT
  else
    G_LOSS=-1; G_RTT=""
  fi
  if [ "$G_LOSS" -ge 0 ] && [ "$G_LOSS" -lt 100 ] && [ "$N_GWOK" != 1 ]; then
    N_GWOK=1; save_net
  fi

  I_LOSS=-1; I_RTT=""; NET_STATUS=-1
  for h in $INET_HOSTS; do
    probe "$h" 3
    if [ "$P_LOSS" -ge 0 ]; then
      any_ping=1
      I_LOSS=$P_LOSS; I_RTT=$P_RTT
      if [ "$P_LOSS" -lt 100 ]; then internet_ping=1; break; fi
    fi
  done
  if [ "$internet_ping" = 1 ]; then
    NET_STATUS=1
  elif https_check; then
    I_LOSS=0; I_RTT=""; NET_STATUS=1
  elif [ "$any_ping" = 1 ]; then
    I_LOSS=100; I_RTT=""; NET_STATUS=0
  else
    # Neither ICMP nor HTTPS provided an answer: do not pretend certainty.
    I_LOSS=-1; I_RTT=""; NET_STATUS=-1
  fi
}
dl_test() {
  local out
  out=$(curl -L -f -sS --connect-timeout 5 --max-time 20 -o /dev/null \
        -w '%{speed_download}' "${SPEED_URL}?bytes=${TEST_BYTES}&r=${RANDOM}${RANDOM}" 2>/dev/null) || return 1
  isnum "$out" || return 1
  awk -v b="$out" 'BEGIN{ if (b<=0) exit 1; printf "%.2f", b*8/1000000 }'
}
battery_low() {
  local out
  command -v termux-battery-status >/dev/null 2>&1 || return 1
  out=$(timeout 8 termux-battery-status 2>/dev/null) || return 1
  printf '%s' "$out" | jq -e '(.percentage < 20) and (.status != "CHARGING") and (.status != "FULL")' >/dev/null 2>&1
}

# ------------------------------ diagnosis
# Sets V_CODE, V_TEXT and V_FIX (whether a reconnect is plausibly useful).
diagnose() {
  V_FIX=1
  local weak=0 stale=0
  if [ "$NET_STATUS" = 0 ]; then
    if [ "$G_LOSS" -ge 0 ] && [ "$G_LOSS" -lt 100 ]; then
      V_CODE=ISP_OR_UPLINK_DOWN; V_FIX=0
      V_TEXT="The router answers, but external connectivity tests fail. This points beyond the phone-to-router Wi-Fi link; reconnecting is unlikely to fix it."
    elif [ "$N_GWOK" = 1 ]; then
      V_CODE=ROUTER_NOT_RESPONDING
      V_TEXT="External connectivity tests fail and the gateway is not answering. A single cautious reconnect may help."
    else
      V_CODE=NO_CONNECTIVITY
      V_TEXT="Connectivity tests fail, but the gateway has not previously been confirmed. A cautious reconnect may help if Android permits it."
    fi
    return
  fi
  if [ "$NET_STATUS" = -1 ] && [ -z "$DL" ]; then
    V_CODE=PROBE_INCONCLUSIVE; V_FIX=0
    V_TEXT="Ping and HTTPS tests could not establish whether the internet is reachable. I will not toggle Wi-Fi on inconclusive evidence."
    return
  fi
  if isint "$W_RSSI" && [ "$W_RSSI" -le "$((0-WEAK_RSSI))" ]; then
    weak=1
  fi
  if [ -n "$W_LINK" ] && isint "$W_RSSI" && [ "$W_RSSI" -gt -70 ] && fge "$N_LINKBEST" 1 \
     && flt "$W_LINK" "$(calc "0.5*$N_LINKBEST")"; then stale=1; fi

  if [ "$weak" = 1 ]; then
    V_CODE=WEAK_SIGNAL; V_FIX=0
    V_TEXT="Signal is weak (${W_RSSI} dBm). Reconnecting will not improve distance or obstacles; move closer to the router or reduce interference."
  elif [ "$stale" = 1 ]; then
    V_CODE=STALE_LINK
    V_TEXT="Signal is reasonable but the Wi-Fi link rate dropped to ${W_LINK} Mbps from a learned best near ${N_LINKBEST} Mbps. Reconnecting may help."
  elif [ "$I_LOSS" -ge 20 ] && [ "$G_LOSS" -ge 20 ]; then
    V_CODE=LOCAL_OR_WAN_LOSS
    V_TEXT="Both gateway and internet probes show loss. Interference, router load, or upstream congestion is possible; a reconnect may help but is not guaranteed."
  elif [ "$I_LOSS" -ge 20 ] && [ "$G_LOSS" -ge 0 ] && [ "$G_LOSS" -lt 20 ]; then
    V_CODE=UPSTREAM_LOSS; V_FIX=0
    V_TEXT="The gateway responds more reliably than external hosts. Upstream congestion or ISP trouble is more likely than a stale phone Wi-Fi link."
  else
    V_CODE=SLOW_UNKNOWN
    V_TEXT="Internet access is present, but the cause of low throughput is uncertain. A reconnect is only a controlled test, not a guaranteed fix."
  fi
}

# ------------------------------ reconnect machinery
pick_method() {
  METHODS=()
  if command -v su >/dev/null 2>&1 && timeout 8 su -c true >/dev/null 2>&1; then METHODS+=(root); fi
  if command -v adb >/dev/null 2>&1; then
    [ -n "${ADB_SERIAL:-}" ] || ADB_SERIAL=$(adb devices 2>/dev/null | awk 'NR>1 && $2=="device" {print $1; exit}')
    if [ -n "${ADB_SERIAL:-}" ] && timeout 8 adb -s "$ADB_SERIAL" get-state >/dev/null 2>&1; then METHODS+=(adb); fi
  fi
  if command -v termux-wifi-enable >/dev/null 2>&1; then METHODS+=(api); fi
}
wifi_set() {
  local v=disable f=false
  if [ "$1" = on ]; then v=enable; f=true; fi
  case "$METHOD" in
    root) timeout 10 su -c "svc wifi $v" >/dev/null 2>&1 ;;
    adb)  timeout 10 adb -s "$ADB_SERIAL" shell "svc wifi $v" >/dev/null 2>&1 ;;
    api)  timeout 10 termux-wifi-enable "$f" >/dev/null 2>&1 ;;
    *)    return 1 ;;
  esac
}
refresh_wifi() {
  local off=$1 ssid0=$W_SSID m t
  RECON=0
  for m in "${METHODS[@]}"; do
    METHOD=$m
    if ! wifi_set off; then continue; fi
    WIFI_OFF=1
    t=0
    while [ "$t" -lt 8 ]; do
      read_wifi || :
      [ "$W_UP" = 1 ] || break
      nap 1; t=$((t + 1))
    done
    if [ "$W_UP" = 1 ]; then
      log "$m method: Android appears to have ignored Wi-Fi switch-off"
      WIFI_OFF=0
      continue
    fi
    nap "$off"
    if wifi_set on || { command -v termux-wifi-enable >/dev/null 2>&1 && termux-wifi-enable true >/dev/null 2>&1; }; then
      WIFI_OFF=1
    else
      log "Wi-Fi enable command failed; cleanup will retry"
      return 1
    fi
    t=0
    while [ "$t" -lt 45 ]; do
      nap 1; t=$((t + 1)); read_wifi || :
      if [ "$W_UP" = 1 ] && same_net "$ssid0"; then WIFI_OFF=0; RECON=$t; return 0; fi
    done
    log "did not rejoin ${ssid0:-the previous network} within 45s"
    return 1
  done
  log "no method could switch Wi-Fi off (tried: ${METHODS[*]:-none})"
  return 1
}
act() {
  local t before after ok=0 att off=2
  t=$(now); diagnose
  if due verdict 60; then log "verdict ${V_CODE}: ${V_TEXT}"; fi
  if [ -f "$PAUSE" ]; then
    bad=0
    due paused 300 && log "paused: monitoring only; Wi-Fi will not be changed"
    return
  fi
  if [ "$t" -lt "$COOL_UNTIL" ]; then bad=0; return; fi
  if [ "$V_FIX" = 0 ]; then
    COOL_UNTIL=$((t + 300)); bad=0
    due advice 900 && notify "Wi-Fi diagnosis" "$V_TEXT"
    return
  fi
  att=$((N_OK + N_FAIL))
  if [ "$att" -ge 4 ] && flt "$(calc "($N_OK+1)/($att+2)")" 0.25; then
    COOL_UNTIL=$((t + MAX_COOL)); bad=0
    if due useless 3600; then
      log "learned: reconnecting rarely helps on ${N_KEY} (${N_OK} of ${att}); standing down"
      notify "Wi-Fi auto-fix standing down" "Reconnects have rarely helped on ${N_KEY}."
    fi
    return
  fi
  pick_method
  if [ "${#METHODS[@]}" -eq 0 ]; then
    COOL_UNTIL=$((t + MAX_COOL)); bad=0
    log "no usable Wi-Fi control method; install/connect ADB or use root if available"
    due recfail 900 && notify "Wi-Fi auto-fix unavailable" "Android has not provided a working Wi-Fi control method. Monitoring continues."
    return
  fi
  [ "$episode_fail" -ge 1 ] && off=6
  before=$DL
  STATE=REFRESHING; live "Wi-Fi: refreshing" "${N_KEY}: ${V_CODE}"
  log "refresh attempt on ${N_KEY}: Wi-Fi off ${off}s (${V_CODE}; method candidates ${METHODS[*]})"
  if refresh_wifi "$off"; then
    nap "$SETTLE"
    after=$(dl_test) || { nap 5; after=$(dl_test) || after=""; }
    threshold
    if [ -n "$after" ] && fge "$after" "$thr"; then
      ok=1
    elif [ -n "$before" ] && [ -n "$after" ] && fge "$after" "$MIN_MBPS" && fge "$after" "$(calc "1.5*$before")"; then
      ok=1
    fi
    if [ "$ok" = 1 ]; then
      N_OK=$((N_OK + 1)); episode_fail=0; cool=$BASE_COOL
      log "refresh helped: ${before:-no prior result} -> ${after} Mbps; rejoined in ${RECON}s"
      notify "Wi-Fi recovery improved speed" "${N_KEY}: ${before:-?} to ${after} Mbps"
      fge "$after" "$thr" && learn "$after"
    else
      N_FAIL=$((N_FAIL + 1)); episode_fail=$((episode_fail + 1))
      cool=$((cool * 2)); [ "$cool" -gt "$MAX_COOL" ] && cool=$MAX_COOL
      log "refresh not confirmed helpful (after ${after:-no result} Mbps); backing off ${cool}s"
    fi
    save_net
    COOL_UNTIL=$(($(now) + cool))
  else
    if [ "$WIFI_OFF" = 1 ]; then
      if wifi_set on || { command -v termux-wifi-enable >/dev/null 2>&1 && termux-wifi-enable true >/dev/null 2>&1; }; then log "issued a final Wi-Fi enable retry"; fi
    fi
    cool=$((cool * 2)); [ "$cool" -gt "$MAX_COOL" ] && cool=$MAX_COOL
    COOL_UNTIL=$(($(now) + cool))
    log "could not complete reconnect; backing off ${cool}s"
    due recfail 900 && notify "Wi-Fi auto-fix failed" "Could not complete the reconnect. Monitoring continues."
  fi
  bad=0
}
count_bad() {
  if [ "$bad" -lt "$NEED" ]; then
    bad=$((bad + 1)); log "$1 (${bad}/${NEED})"
  elif due stillbad 120; then
    log "$1 (already confirmed; waiting for next permitted check)"
  fi
}
evaluate() {
  threshold
  log "${N_KEY} $(band "$W_FREQ") ${W_RSSI:-?}dBm link ${W_LINK:-?} | router ${G_RTT:--}ms loss ${G_LOSS}% | internet ${I_RTT:--}ms loss ${I_LOSS}% | ${DL:-failed} Mbps (normal ${N_BASE}, alert <${thr}; connectivity ${NET_STATUS})"
  if [ -z "$DL" ]; then
    if [ "$NET_STATUS" = 0 ]; then
      count_bad "internet connectivity test failed"
    else
      log "speed-test endpoint failed but internet status is not confirmed down; ignoring this sample"
    fi
  elif flt "$DL" "$thr"; then
    count_bad "slow: ${DL} Mbps < ${thr}"
  else
    [ "$bad" -gt 0 ] && log "back to normal without a reconnect"
    bad=0; episode_fail=0; cool=$BASE_COOL
    learn "$DL"
  fi
  if [ "$bad" -ge "$NEED" ]; then act; fi
  if [ "$(now)" -lt "$COOL_UNTIL" ]; then STATE=COOLDOWN
  elif [ "$bad" -gt 0 ]; then STATE=WATCHING
  else STATE=OK; fi
  live "Wi-Fi ${STATE}" "${N_KEY}: ${DL:-no result} Mbps (normal ${N_BASE})"
}
write_last() {
  printf '%s  %s (%s)  %s dBm  link %s Mbps  router %s ms  internet %s ms  last test %s Mbps (normal %s)  state %s\n' \
    "$(date +%T)" "$N_KEY" "$(band "$W_FREQ")" "${W_RSSI:-?}" "${W_LINK:-?}" "${G_RTT:--}" "${I_RTT:--}" \
    "$LAST_DL" "$N_BASE" "$STATE" >"$LAST"
}

cleanup() {
  trap - EXIT
  if [ "${WIFI_OFF:-0}" = 1 ]; then
    wifi_set on || { command -v termux-wifi-enable >/dev/null 2>&1 && termux-wifi-enable true >/dev/null 2>&1; }
  fi
  command -v termux-wake-unlock >/dev/null 2>&1 && termux-wake-unlock >/dev/null 2>&1
  command -v termux-notification-remove >/dev/null 2>&1 && termux-notification-remove wififix_live >/dev/null 2>&1
  [ "$(cat "$PIDF" 2>/dev/null)" = "$$" ] && rm -f "$PIDF"
}
engine() {
  local last_full=0 last_batt=0 t gap susp susp_seen=0 key cur_key="" cyc
  need jq jq; need curl curl; need termux-wifi-connectioninfo termux-api
  if running; then echo "Already running (pid $(cat "$PIDF"))."; exit 1; fi
  echo $$ >"$PIDF"; chmod 600 "$PIDF"
  trap cleanup EXIT
  trap 'exit 0' INT TERM HUP
  command -v termux-wake-lock >/dev/null 2>&1 && termux-wake-lock >/dev/null 2>&1
  pick_method
  log "engine started; Wi-Fi control methods: ${METHODS[*]:-none}; floor ${MIN_MBPS} Mbps, adaptive ${ADAPT_FRAC}x normal, heartbeat ${HEARTBEAT}s"
  if ! command -v ip >/dev/null 2>&1; then log "ip command unavailable; gateway lookup may use /proc/net/route. Install iproute2 for best diagnostics."; fi

  while :; do
    t=$(now)
    if [ $((t - last_batt)) -ge 300 ]; then
      last_batt=$t
      if battery_low; then LOWPOWER=1; else LOWPOWER=0; fi
    fi
    cyc=$CYCLE; [ "$LOWPOWER" = 1 ] && cyc=$((CYCLE * 3))
    read_wifi || :
    if [ "$W_UP" != 1 ]; then
      due offline 120 && log "Wi-Fi not connected/readable; waiting without toggling it"
      nap "$cyc"; continue
    fi
    key=$(net_key)
    if [ "$key" != "$cur_key" ]; then
      cur_key=$key; load_net "$key"; bad=0; episode_fail=0; last_full=0
      log "on ${key} ($(band "$W_FREQ")); learned normal ${N_BASE} Mbps from ${N_SAMP} tests"
      [ "$key" = unknown ] && log "Wi-Fi name hidden; grant Termux:API the required permission if you want separate profiles by network"
    fi
    cheap_probe
    susp=0
    [ "$NET_STATUS" = 0 ] && susp=1
    [ "$I_LOSS" -ge 20 ] && susp=1
    [ "$G_LOSS" -ge 20 ] && [ "$I_LOSS" -ge 20 ] && susp=1
    if isint "$W_RSSI" && [ "$W_RSSI" -le "$((0-WEAK_RSSI))" ]; then susp=1; fi
    if [ -n "$W_LINK" ] && fge "$N_LINKBEST" 1 && flt "$W_LINK" "$(calc "0.5*$N_LINKBEST")"; then susp=1; fi
    if [ -n "$I_RTT" ] && fge "$N_RTT" 1 && fge "$I_RTT" "$(calc "3*$N_RTT+15")"; then susp=1; fi
    [ "$susp" = 0 ] && susp_seen=0
    gap=$HEARTBEAT
    if [ "$bad" -gt 0 ]; then gap=$SUSPECT_GAP
    elif [ "$susp" = 1 ] && [ "$susp_seen" = 0 ]; then gap=$SUSPECT_GAP; fi
    if [ "$(now)" -lt "$COOL_UNTIL" ] && [ "$gap" -lt $((HEARTBEAT / 2)) ]; then gap=$((HEARTBEAT / 2)); fi
    DL=""
    if [ $((t - last_full)) -ge "$gap" ]; then
      DL=$(dl_test) || DL=""
      last_full=$(now)
      [ -n "$DL" ] && LAST_DL=$DL
      [ "$susp" = 1 ] && susp_seen=1
      evaluate
    fi
    write_last
    nap "$cyc"
  done
}

cmd_diagnose() {
  need jq jq; need curl curl; need termux-wifi-connectioninfo termux-api
  read_wifi || :
  if [ "$W_UP" != 1 ]; then echo "Wi-Fi is not connected/readable."; exit 1; fi
  load_net "$(net_key)"
  echo "Running connectivity and speed checks; this may use up to ${TEST_BYTES} bytes for the download test."
  cheap_probe
  DL=$(dl_test) || DL=""
  threshold
  if [ -n "$DL" ] && ! flt "$DL" "$thr" && [ "$NET_STATUS" = 1 ] && [ "$I_LOSS" -lt 20 ]; then
    V_CODE=HEALTHY; V_FIX=0; V_TEXT="Connectivity and throughput look healthy for this network."
  else
    diagnose
  fi
  printf 'Network      : %s (%s)\n' "$N_KEY" "$(band "$W_FREQ")"
  printf 'Signal       : %s dBm\n' "${W_RSSI:-unknown}"
  printf 'Link rate    : %s Mbps (best seen %s)\n' "${W_LINK:-unknown}" "$N_LINKBEST"
  printf 'Gateway      : %s, %s ms, %s%% loss\n' "$(gw_addr 2>/dev/null || echo unknown)" "${G_RTT:--}" "$G_LOSS"
  printf 'Internet     : status %s, %s ms, %s%% loss\n' "$NET_STATUS" "${I_RTT:--}" "$I_LOSS"
  printf 'Download     : %s Mbps (normal %s, alert below %s)\n' "${DL:-failed}" "$N_BASE" "$thr"
  printf '\nVerdict      : %s\n%s\n' "$V_CODE" "$V_TEXT"
  if [ "${#METHODS[@]}" -eq 0 ]; then :; fi
}
cmd_stats() {
  local f any=0
  for f in "$D"/nets/*; do
    [ -f "$f" ] || continue
    any=1
    # Parse using the safe loader rather than sourcing files.
    local oldnf="$NF" oldkey="$N_KEY"
    N_KEY="$(basename "$f")"; NF="$f"
    N_BASE=0; N_SAMP=0; N_LINKBEST=0; N_OK=0; N_FAIL=0; N_RTT=0; N_GWOK=0
    local key value
    while IFS='=' read -r key value; do
      case "$key" in
        N_BASE|N_LINKBEST|N_RTT) if isnum "$value"; then printf -v "$key" '%s' "$value"; fi ;;
        N_SAMP|N_OK|N_FAIL) if isint "$value" && [ "$value" -ge 0 ]; then printf -v "$key" '%s' "$value"; fi ;;
        N_GWOK) case "$value" in 0|1) N_GWOK=$value ;; esac ;;
      esac
    done < "$f"
    printf '%s\n  normal %s Mbps over %s tests, best link %s Mbps\n  reconnects: %s helped, %s did not\n' \
      "$(sed -n '1s/^# SSID: //p' "$f")" "$N_BASE" "$N_SAMP" "$N_LINKBEST" "$N_OK" "$N_FAIL"
    NF="$oldnf"; N_KEY="$oldkey"
  done
  [ "$any" = 1 ] || echo "Nothing learned yet. Start the engine and allow a few good speed tests."
}
cmd_status() {
  if running; then echo "Running (pid $(cat "$PIDF"))."; else echo "Not running."; fi
  [ -f "$PAUSE" ] && echo "Paused: monitoring only."
  [ -f "$LAST" ] && cat "$LAST"
  [ -f "$LOG" ] && { echo "--- recent events ---"; tail -n 8 "$LOG"; }
  return 0
}

case "${1:-run}" in
  run) engine ;;
  start)
    if running; then echo "Already running (pid $(cat "$PIDF"))."
    else nohup bash "$SELF" run >/dev/null 2>&1 & sleep 2
      if running; then echo "Started (pid $(cat "$PIDF")). Check with: bash $SELF status"
      else echo "Could not confirm startup. Try: bash $SELF run"; fi
    fi ;;
  stop) if running; then kill "$(cat "$PIDF")"; echo "Stopping."; else echo "Not running."; fi ;;
  status) cmd_status ;;
  stats) cmd_stats ;;
  diagnose) cmd_diagnose ;;
  pause) touch "$PAUSE"; chmod 600 "$PAUSE"; echo "Paused. Monitoring continues, but the engine will not change Wi-Fi." ;;
  resume) rm -f "$PAUSE"; echo "Resumed." ;;
  *) echo "Usage: $0 [run|start|stop|status|stats|diagnose|pause|resume]"; exit 1 ;;
esac
