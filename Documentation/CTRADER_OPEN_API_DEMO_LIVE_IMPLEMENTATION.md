# cTrader Open API — Dual Connection (Demo + Live) Implementation Guide

**Purpose**: Provide a robust, institutional-grade reference for running **two simultaneous cTrader Open API connections** (one demo, one live) for trading, while preserving operational safety, auditability, and isolation.

**Scope**: Authentication, connectivity, environment isolation, runtime guardrails, SDK templates, and promotion runbook.

---

## 1) Core Constraints (Non‑Negotiable)

- **Demo and live environments are fully separated**; always use **two independent connections** if you need both at the same time.
- **Do not mix credentials, tokens, or account IDs** between environments.
- **Heartbeat required**: one keep‑alive per connection on the interval specified by the Open API.
- **Rate limits** apply **per connection**; throttle or queue on breaches.

---

## 2) Endpoints & Protocol Selection

**Two connections, two endpoints** (one demo, one live). You can choose **Protobuf** or **JSON** but must keep the same protocol per connection.

| Environment | Protobuf | JSON |
|---|---|---|
| Live | `live.ctraderapi.com:5035` | `live.ctraderapi.com:5036` |
| Demo | `demo.ctraderapi.com:5035` | `demo.ctraderapi.com:5036` |

**Policy**: Default to **Protobuf** for production stability and SDK support unless you have a JSON‑only requirement.

---

## 3) Authentication Flow (Per Environment)

For each environment, run the full OAuth + Open API flow:

1. **OAuth authorization** → obtain `code`
2. **Exchange `code` for tokens** → receive `accessToken` (+ `refreshToken`)
3. **Application auth** → `ProtoOAApplicationAuthReq` (clientId + clientSecret)
4. **Account discovery** → `ProtoOAGetAccountListByAccessTokenReq`
5. **Account auth** → `ProtoOAAccountAuthReq` (ctidTraderAccountId + accessToken)

**Notes**
- Store **tokens separately per environment**.
- Ensure **account ID is numeric** (ctidTraderAccountId).
- Do not send trading requests before app + account auth succeeds.

---

## 4) Environment Variables (Aligned to Current `.env`)

Per `SECURITY.md` and the security audit guides, **do not commit secrets** and **never echo credentials in logs**. This document uses placeholders only.

### Current `.env` Keys (as implemented)

These keys exist in the repository `.env` and are read by the current client (`execution/ctrader_client.py`):

```
EMAIL_CTRADER=...
PASSWORD_CTRADER=...
APPLICATION_NAME_CTRADER=...
USERNAME_CTRADER=...
CTRADER_ACCOUNT_ID=...
CTRADER_APPLICATION_SECRET=...
CTRADER_APPLICATION_ID=...
CTRADER_USERNAME=...
CTRADER_PASSWORD=...
CTRADER_EMAIL=...
CTRADER_URL=...
CTRADER_URL_LOGIN=...
CTRADER_URL_LOGOUT=...
CTRADER_URL_REFRESH=...
CTRADER_URL_VERIFY=...
```

**Resolution order in code** (simplified):
`USERNAME_CTRADER/CTRADER_USERNAME` → `EMAIL_CTRADER/CTRADER_EMAIL` (fallback)
`PASSWORD_CTRADER/CTRADER_PASSWORD`
`APPLICATION_NAME_CTRADER/CTRADER_APPLICATION_ID`
`CTRADER_APPLICATION_SECRET`
`CTRADER_ACCOUNT_ID` (must coerce to integer)

### Recommended Dual‑Environment Extension (Optional)

To run **demo + live simultaneously**, keep the existing keys for demo and add explicit live keys (do **not** replace the current ones unless you also update the loader):

```
CTRADER_DEMO_USERNAME=...
CTRADER_DEMO_PASSWORD=...
CTRADER_DEMO_APPLICATION_ID=...
CTRADER_DEMO_APPLICATION_SECRET=...
CTRADER_DEMO_ACCOUNT_ID=...

CTRADER_LIVE_USERNAME=...
CTRADER_LIVE_PASSWORD=...
CTRADER_LIVE_APPLICATION_ID=...
CTRADER_LIVE_APPLICATION_SECRET=...
CTRADER_LIVE_ACCOUNT_ID=...

CTRADER_LIVE_ENABLED=false
```

**Important**: If you introduce the dual‑env keys, also update the loader logic to read them explicitly for each connection. Do **not** reuse a single set of credentials across demo and live connections.

---

## 5) SDK‑Specific Code Templates (Python)

> These are **templates** aligned with the cTrader Open API SDK conventions. Adjust imports/names to match the SDK version you install.

### 5.1 Dual Connection Skeleton (Demo + Live)

```python
from ctrader_open_api import Client, Protobuf
from ctrader_open_api.endpoints import Endpoints
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoHeartbeatEvent
from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAApplicationAuthReq,
    ProtoOAGetAccountListByAccessTokenReq,
    ProtoOAAccountAuthReq,
)

DEMO_ENDPOINT = Endpoints.DEMO  # demo.ctraderapi.com:5035
LIVE_ENDPOINT = Endpoints.LIVE  # live.ctraderapi.com:5035

class DualCtrader:
    def __init__(self, demo_cfg, live_cfg):
        self.demo = Client(DEMO_ENDPOINT)
        self.live = Client(LIVE_ENDPOINT)
        self.demo_cfg = demo_cfg
        self.live_cfg = live_cfg

    def connect(self):
        self.demo.start()
        self.live.start()

    def auth_demo(self):
        self._auth(self.demo, self.demo_cfg)

    def auth_live(self):
        self._auth(self.live, self.live_cfg)

    def _auth(self, client, cfg):
        client.send(ProtoOAApplicationAuthReq(
            clientId=cfg.client_id,
            clientSecret=cfg.client_secret,
        ))
        client.send(ProtoOAGetAccountListByAccessTokenReq(
            accessToken=cfg.access_token,
        ))
        client.send(ProtoOAAccountAuthReq(
            ctidTraderAccountId=cfg.account_id,
            accessToken=cfg.access_token,
        ))

    def heartbeat(self, client):
        client.send(ProtoHeartbeatEvent())
```

### 5.2 Safe‑Mode Order Routing (Explicit Environment)

```python
def route_order(env: str, order_req, dual: DualCtrader):
    if env == "live":
        if not dual.live_cfg.live_enabled:
            raise RuntimeError("Live trading disabled by flag")
        dual.live.send(order_req)
    else:
        dual.demo.send(order_req)
```

### 5.3 Robust Token Refresh (Pseudo‑Flow)

```python
if token_expires_soon:
    refresh_access_token(refresh_token)
    re_auth_application()
    re_auth_account()
```

---

## 6) Runbook — Demo → Live Promotion

### Phase A — Demo‑Only Validation
- [ ] App registered and OAuth flow verified
- [ ] Demo account list returned
- [ ] Demo account auth succeeds
- [ ] Heartbeat loop stable for 30+ minutes
- [ ] Place/modify/cancel test orders in demo
- [ ] Orders + positions logged to audit files

### Phase B — Live Safe‑Mode (Read‑Only)
- [ ] Live connection established
- [ ] Live auth success
- [ ] Live **read‑only** queries (positions, account info)
- [ ] No orders allowed (`CTRADER_LIVE_ENABLED=false`)

### Phase C — Live Trading (Explicit Enable)
- [ ] Enable: `CTRADER_LIVE_ENABLED=true`
- [ ] Confirm risk flags and guardrails
- [ ] Place minimal test order (micro‑size)
- [ ] Verify full audit chain: order → position → P&L

### Phase D — Post‑Promotion Monitoring
- [ ] Ensure order and position logs are written for every trade
- [ ] Run daily position snapshot
- [ ] Verify slippage + execution cost logs
- [ ] Require rollback readiness before scaling size

---

## 7) Failure Handling (Production‑Safe Defaults)

- Heartbeat failure → backoff + reconnect
- Auth failure → disable trading on that environment
- Rate limit breach → throttle or queue
- Connection drop → re‑query open positions before resuming

---

## 8) Audit Artifacts (Required)

- `logs/ctrader/demo_connection.log`
- `logs/ctrader/live_connection.log`
- `logs/ctrader/order_audit.jsonl`
- `logs/ctrader/position_snapshot.jsonl`

---

## 9) Success Criteria

- Demo and live connections run concurrently without cross‑env leakage
- Demo and live orders are recorded with explicit environment tags
- Live can be toggled on/off with a single flag
- All audit logs are written for each trade

---

## 10) Notes for This Repo

- The project already ships `execution/ctrader_client.py` (REST‑style). Use it as the baseline until SDK wiring is required.
- If you adopt the SDK, keep a **single adapter** layer that conforms to the current `paper_trading_engine`/order routing contract.

---

## 11) References (Official)

- cTrader Open API documentation (Open API overview, endpoints, authentication, heartbeat, and SDK).
