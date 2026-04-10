# Service Setup

Motorcade runs four persistent services: three specialist sidecars and a dashboard. They use two different process models.

## Process Models

### systemd (root-level units in `/etc/systemd/system/`)

Used for the FED sidecar and the vertical dashboard. Services survive logout, restart automatically on failure, and log to `var/logs/` via `StandardOutput=append:...`.

```bash
sudo systemctl start <service>
sudo systemctl stop <service>
sudo systemctl enable <service>   # start on boot
sudo systemctl status <service>
```

### nohup / manual foreground

Used for weather and crypto. Started manually against a terminal session; output goes to `nohup.out` (weather) or `var/logs/crypto_sidecar.log` (crypto). Not restart-on-failure, not boot-persistent.

```bash
cd sidecars/weather && nohup ./start.sh &
cd sidecars/crypto  && nohup ./start.sh &
```

---

## Services

### Weather sidecar

| | |
|---|---|
| **URL** | `http://127.0.0.1:8765` (env: `WEATHER_SPECIALIST_URL`) |
| **Start script** | `sidecars/weather/start.sh` |
| **Process model** | `nohup` — manual, not boot-persistent |
| **Log** | `sidecars/weather/nohup.out` |
| **Covers** | `KXHIGHT{BOS,DAL,HOU,SEA,PHX,SATX,LV,ATL,MIN,NOLA,DC,SFO,OKC}` |

The start script just `exec`s `sidecar.py` from the sidecar's venv with no extra env sourcing. Port is hardcoded to 8765 in the sidecar; `WEATHER_SIDECAR_HOST/PORT` env vars are referenced in the script comment but not currently wired through to uvicorn args.

### Crypto sidecar

| | |
|---|---|
| **URL** | `http://127.0.0.1:8766` (env: `CRYPTO_SPECIALIST_URL`) |
| **Start script** | `sidecars/crypto/start.sh` |
| **Process model** | `nohup` — manual, not boot-persistent |
| **Log** | `sidecars/crypto/var/logs/crypto_sidecar.log` |
| **Covers** | `KXBTCD-*`, `KXETHD-*`, `KXSOLD-*`, `KXXRPD-*` |

Same pattern as weather: a thin wrapper that `exec`s `sidecar.py`. Crypto maintains its own log directory under `sidecars/crypto/var/logs/`.

### FED sidecar (hawkwatchers)

| | |
|---|---|
| **URL** | `http://127.0.0.1:8768` (env: `FED_SPECIALIST_URL`) |
| **Start script** | `sidecars/hawkwatchers/start_fed_sidecar.sh` |
| **Process model** | systemd — `fed-sidecar.service` |
| **Unit file** | `/etc/systemd/system/fed-sidecar.service` |
| **Log** | `var/logs/fed_sidecar.log` |
| **Covers** | `KXFED-*`, `KXFOMC-*` |

The start script sources `.env` (if present), checks for `models/best_clf.joblib` and runs `train.py` if missing, then starts uvicorn. The systemd unit has `Restart=on-failure` with a 10s delay but is currently **disabled** (not started on boot). Status as of 2026-04-10: **inactive (dead)**.

To start:
```bash
sudo systemctl start fed-sidecar.service
sudo systemctl enable fed-sidecar.service  # if boot-persistence is wanted
```

### Vertical dashboard

| | |
|---|---|
| **URL** | `http://127.0.0.1:8181` (env: `VERTICAL_DASHBOARD_PORT`) |
| **Start script** | `scripts/start_vertical_dashboard.sh` |
| **Process model** | systemd — `vertical-dashboard.service` |
| **Unit file** | `/etc/systemd/system/vertical-dashboard.service` |
| **Serves** | `vertical_dashboard.html`, rebuilt from `var/research/` every 60s |

The systemd unit hardcodes key env vars (`BOT_RESEARCH_DIR`, port, refresh interval, bot log path/lines) directly in the unit file rather than sourcing `.env`. `Restart=always`.

---

## Summary Table

| Service | Port | Process model | Auto-restart | Boot-persistent | Status (2026-04-10) |
|---|---|---|---|---|---|
| weather | 8765 | nohup | no | no | running |
| crypto | 8766 | nohup | no | no | running |
| fed (hawkwatchers) | 8768 | systemd | yes (on-failure) | no (disabled) | dead |
| vertical-dashboard | 8181 | systemd | yes (always) | yes (enabled) | running |

---

## Gaps

- Weather and crypto have no automatic restart or boot persistence. A server reboot or terminal close will kill them.
- The FED sidecar unit exists but is disabled. It should be enabled once the model is confirmed trained.
- Standardizing weather and crypto onto systemd units (following the fed pattern) would make the full set consistent.
