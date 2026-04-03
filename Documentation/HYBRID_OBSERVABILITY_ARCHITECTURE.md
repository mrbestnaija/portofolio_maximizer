# Hybrid Observability Architecture

## Goal

Use the static dashboard as the **canonical run/evidence view**, add
**Prometheus as the alerting layer**, and defer **Grafana/Loki** until the
system is operating like a true multi-service production platform.

This repo already has strong artifact-centric semantics. The right next step is
to preserve that strength while adding low-friction alerting.

## Decision

### Keep as canonical

- `visualizations/live_dashboard.html`
- `scripts/dashboard_db_bridge.py`
- `visualizations/dashboard_data.json`
- `data/dashboard_audit.db`
- `logs/audit_gate/production_gate_latest.json`

These artifacts remain the operator-facing and audit-facing truth.

### Add now

- `scripts/prometheus_alert_exporter.py`
- `config/prometheus.yml`
- `config/prometheus_alert_rules.yml`

These are **alerting surfaces only**. They should never replace the evidence
artifacts above.

### Defer

- Grafana dashboards
- Loki / promtail log shipping
- Centralized cross-host log search
- Full docker-compose observability stack

Those are useful later, but they are not the immediate bottleneck.

## Architecture

```text
SQLite DB + audit artifacts
    |
    |  read-only / artifact-driven
    v
scripts/dashboard_db_bridge.py
    |
    +--> visualizations/dashboard_data.json
    |        |
    |        v
    |    visualizations/live_dashboard.html
    |    Canonical human run/evidence view
    |
    +--> data/dashboard_audit.db
             Snapshot persistence / replay audit

logs/audit_gate/production_gate_latest.json
    |
    +--> live_dashboard evidence panel
    +--> Prometheus exporter metrics

scripts/prometheus_alert_exporter.py
    |
    v
Prometheus
    |
    v
Alert rules only

Later:
Prometheus + Grafana + Loki + promtail once PMX is multi-host / multi-service
```

## Responsibilities

### Static Dashboard

The static dashboard is the canonical view because it can display:

- latest-run filtering
- positions and trade events
- price series with trade markers
- artifact binding and proof runway context
- provenance and synthetic/live origin
- persisted dashboard audit snapshot state

Prometheus is not a good replacement for that information density.

### Prometheus

Prometheus should answer narrow operational questions:

- Is the canonical payload missing?
- Is the payload stale?
- Is the production gate passing?
- Is the production gate artifact stale under the same freshness policy as the canonical dashboard?
- Is artifact binding broken?
- Is the proof runway incomplete?
- Are audit snapshots missing?
- Is the audit snapshot DB unreadable/corrupt?
- Is robustness degraded?
- Is the payload origin non-live?

That is enough to page or warn, without creating a second human-facing
dashboard that drifts from the evidence view.

## Artifact Contract

Prometheus reads only the canonical artifacts below:

- `visualizations/dashboard_data.json`
- `logs/audit_gate/production_gate_latest.json`
- `data/dashboard_audit.db`

It does **not** compute business truth independently.

## Operational Modes

### Now

- Single-machine / local-first
- Static HTML evidence dashboard
- Read-only DB bridge
- Prometheus scrape target on `127.0.0.1:9108`
- Alert rules driven by artifact freshness and gate status
- Prometheus freshness thresholds reuse the same env-driven policies as the canonical dashboard bridge

### Later

Adopt Grafana/Loki only when the repo has:

- multiple long-running services
- shared operators/on-call
- central log retention needs
- cross-host correlation requirements
- stable metrics taxonomy worth dashboarding beyond alerts

## Commands

Run the canonical dashboard stack:

```powershell
python -m scripts.dashboard_db_bridge --persist-snapshot
python -m http.server 8000 --bind 127.0.0.1 --directory .
```

Windows one-step launch with refresh + the human-facing localhost dashboard stack:

```powershell
python scripts/windows_dashboard_manager.py launch
```

or:

```powershell
.\launch_live_dashboard.bat
```

This refreshes `visualizations/dashboard_data.json`, forces a fresh `logs/audit_gate/production_gate_latest.json` once during launch, starts the bridge, local HTTP server, and Prometheus alert exporter, and opens the dashboard. The running bridge keeps the production gate artifact fresh when it becomes stale, and the live watcher remains opt-in via `--ensure-live-watcher`.

The steady-state payload preserves the last successful refresh actor and timestamp for the current gate artifact, so the Evidence Chain can show both `gate_refresh=skipped/fresh_artifact` and that `dashboard_launch` was the actor that most recently refreshed this exact artifact.

Open:

```text
http://127.0.0.1:8000/visualizations/live_dashboard.html
```

Run the Prometheus exporter:

```powershell
python scripts/prometheus_alert_exporter.py --listen-host 127.0.0.1 --port 9108
```

Print metrics once:

```powershell
python scripts/prometheus_alert_exporter.py --once
```

## Why This Split Fits PMX

PMX is currently evidence-constrained more than infrastructure-constrained.

The production gate, proof runway, artifact binding, and canonical payload
freshness are the most important truths to preserve. The static dashboard is
already wired around those concepts. Prometheus should strengthen alerting
around them, not replace them.
