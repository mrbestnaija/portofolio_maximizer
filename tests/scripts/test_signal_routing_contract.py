from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from models.signal_router import validate_routing_contract


ROOT = Path(__file__).resolve().parents[2]


def test_signal_routing_config_surfaces_noop_knobs() -> None:
    cfg_path = ROOT / "config" / "signal_routing_config.yml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    routing = raw.get("signal_routing") or {}
    warnings = validate_routing_contract(routing, strict=False)
    assert "unsupported_routing_knob:enable_samossa" in warnings
    assert "unsupported_routing_knob:enable_sarimax" in warnings
    assert "unsupported_routing_knob:enable_garch" in warnings
    assert "unsupported_routing_knob:enable_mssa_rl" in warnings
    assert "unsupported_routing_knob:routing_mode" in warnings


def test_strict_routing_contract_rejects_noop_knobs() -> None:
    with pytest.raises(ValueError):
        validate_routing_contract({"enable_samossa": True}, strict=True)
