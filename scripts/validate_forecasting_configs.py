#!/usr/bin/env python3
"""Validate forecasting/pipeline configuration integrity.

This guard ensures:
1) forecasting config contains required sections and ensemble schema.
2) pipeline config embeds a compatible forecasting block.
3) candidate weight dictionaries are structurally valid.
4) regime candidate weights avoid disabled SARIMAX contamination.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORECASTING_PATH = ROOT / "config" / "forecasting_config.yml"
DEFAULT_PIPELINE_PATH = ROOT / "config" / "pipeline_config.yml"

REQUIRED_FORECASTING_SECTIONS = (
    "sarimax",
    "garch",
    "samossa",
    "mssa_rl",
    "ensemble",
    "regime_detection",
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML object: {path}")
    return payload


def _extract_forecasting(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    if "pipeline" in payload and isinstance(payload.get("pipeline"), dict):
        return payload["pipeline"].get("forecasting") or {}
    return payload.get("forecasting") or payload


def _validate_candidate_weights(
    *,
    candidate_weights: Any,
    min_component_weight: float,
    context: str,
    disallowed_models: Optional[set[str]] = None,
) -> List[str]:
    errors: List[str] = []
    if not isinstance(candidate_weights, list) or not candidate_weights:
        return [f"{context}: ensemble.candidate_weights must be a non-empty list"]

    for idx, candidate in enumerate(candidate_weights):
        if not isinstance(candidate, dict) or not candidate:
            errors.append(f"{context}: candidate_weights[{idx}] must be a non-empty mapping")
            continue
        total = 0.0
        for model, weight in candidate.items():
            if not isinstance(model, str):
                errors.append(f"{context}: candidate_weights[{idx}] has non-string model key")
                continue
            model_key = model.strip().lower()
            if disallowed_models and model_key in disallowed_models:
                errors.append(
                    f"{context}: candidate_weights[{idx}] includes disabled '{model_key}'"
                )
            try:
                w = float(weight)
            except (TypeError, ValueError):
                errors.append(f"{context}: candidate_weights[{idx}]['{model}'] must be numeric")
                continue
            if w <= 0:
                errors.append(
                    f"{context}: candidate_weights[{idx}]['{model}'] must be > 0 (got {w})"
                )
            if w < min_component_weight:
                errors.append(
                    f"{context}: candidate_weights[{idx}]['{model}']={w} "
                    f"< minimum_component_weight={min_component_weight}"
                )
            total += w
        if abs(total - 1.0) > 0.01:
            errors.append(
                f"{context}: candidate_weights[{idx}] sums to {total:.6f}, expected ~1.0"
            )
    return errors


def _validate_regime_candidate_weights(regime_cfg: Dict[str, Any], context: str) -> List[str]:
    errors: List[str] = []
    rcw = regime_cfg.get("regime_candidate_weights")
    if rcw is None:
        return errors
    if not isinstance(rcw, dict):
        return [f"{context}: regime_detection.regime_candidate_weights must be a mapping"]

    for regime, candidates in rcw.items():
        if not isinstance(candidates, list):
            errors.append(
                f"{context}: regime_candidate_weights['{regime}'] must be a list of candidates"
            )
            continue
        for idx, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                errors.append(
                    f"{context}: regime_candidate_weights['{regime}'][{idx}] must be a mapping"
                )
                continue
            if "sarimax" in candidate:
                errors.append(
                    f"{context}: regime_candidate_weights['{regime}'][{idx}] includes disabled 'sarimax'"
                )
    return errors


def _validate_required_sections(cfg: Dict[str, Any], context: str) -> List[str]:
    errors: List[str] = []
    for section in REQUIRED_FORECASTING_SECTIONS:
        if section not in cfg:
            errors.append(f"{context}: missing required section '{section}'")
    return errors


def _validate_sarimax_disabled(cfg: Dict[str, Any], context: str) -> List[str]:
    sarimax = cfg.get("sarimax") if isinstance(cfg, dict) else None
    if not isinstance(sarimax, dict):
        return [f"{context}: section 'sarimax' must be a mapping"]
    if sarimax.get("enabled") is not False:
        return [f"{context}: sarimax.enabled must be false by default"]
    return []


def _sync_checks(
    forecasting_cfg: Dict[str, Any],
    pipeline_cfg: Dict[str, Any],
) -> List[str]:
    errors: List[str] = []
    f_sarimax = ((forecasting_cfg.get("sarimax") or {}).get("enabled"))
    p_sarimax = ((pipeline_cfg.get("sarimax") or {}).get("enabled"))
    if f_sarimax != p_sarimax:
        errors.append(
            "sync: sarimax.enabled mismatch between forecasting and pipeline configs"
        )

    f_candidates = ((forecasting_cfg.get("ensemble") or {}).get("candidate_weights") or [])
    p_candidates = ((pipeline_cfg.get("ensemble") or {}).get("candidate_weights") or [])
    if f_candidates != p_candidates:
        errors.append(
            "sync: ensemble.candidate_weights mismatch between forecasting and pipeline configs"
        )
    return errors


def validate_configs(
    forecasting_config_path: Path,
    pipeline_config_path: Path,
) -> Dict[str, Any]:
    forecast_raw = _load_yaml(forecasting_config_path)
    pipeline_raw = _load_yaml(pipeline_config_path)

    forecast_cfg = _extract_forecasting(forecast_raw)
    pipeline_cfg = _extract_forecasting(pipeline_raw)

    errors: List[str] = []
    warnings: List[str] = []

    def _disabled_models(cfg: Dict[str, Any]) -> set[str]:
        disabled: set[str] = set()
        for model in ("sarimax", "garch", "samossa", "mssa_rl"):
            node = cfg.get(model) if isinstance(cfg, dict) else None
            if isinstance(node, dict) and node.get("enabled") is False:
                disabled.add(model)
        return disabled

    errors.extend(_validate_required_sections(forecast_cfg, "forecasting_config"))
    errors.extend(_validate_required_sections(pipeline_cfg, "pipeline_config"))
    errors.extend(_validate_sarimax_disabled(forecast_cfg, "forecasting_config"))
    errors.extend(_validate_sarimax_disabled(pipeline_cfg, "pipeline_config"))

    ensemble = forecast_cfg.get("ensemble") if isinstance(forecast_cfg, dict) else {}
    if isinstance(ensemble, dict):
        min_component_weight = float(ensemble.get("minimum_component_weight", 0.05))
        errors.extend(
            _validate_candidate_weights(
                candidate_weights=ensemble.get("candidate_weights"),
                min_component_weight=min_component_weight,
                context="forecasting_config",
                disallowed_models=_disabled_models(forecast_cfg),
            )
        )
    else:
        errors.append("forecasting_config: section 'ensemble' must be a mapping")

    p_ensemble = pipeline_cfg.get("ensemble") if isinstance(pipeline_cfg, dict) else {}
    if isinstance(p_ensemble, dict):
        p_min_component_weight = float(p_ensemble.get("minimum_component_weight", 0.05))
        errors.extend(
            _validate_candidate_weights(
                candidate_weights=p_ensemble.get("candidate_weights"),
                min_component_weight=p_min_component_weight,
                context="pipeline_config",
                disallowed_models=_disabled_models(pipeline_cfg),
            )
        )
    else:
        errors.append("pipeline_config: section 'ensemble' must be a mapping")

    regime_cfg = forecast_cfg.get("regime_detection") if isinstance(forecast_cfg, dict) else {}
    if isinstance(regime_cfg, dict):
        errors.extend(_validate_regime_candidate_weights(regime_cfg, "forecasting_config"))
    else:
        errors.append("forecasting_config: section 'regime_detection' must be a mapping")

    p_regime_cfg = pipeline_cfg.get("regime_detection") if isinstance(pipeline_cfg, dict) else {}
    if isinstance(p_regime_cfg, dict):
        errors.extend(_validate_regime_candidate_weights(p_regime_cfg, "pipeline_config"))
    else:
        errors.append("pipeline_config: section 'regime_detection' must be a mapping")

    errors.extend(_sync_checks(forecast_cfg, pipeline_cfg))

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "forecasting_config_path": str(forecasting_config_path),
        "pipeline_config_path": str(pipeline_config_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate forecasting/pipeline config integrity.")
    parser.add_argument(
        "--forecasting-config",
        default=str(DEFAULT_FORECASTING_PATH),
        help="Path to forecasting_config.yml",
    )
    parser.add_argument(
        "--pipeline-config",
        default=str(DEFAULT_PIPELINE_PATH),
        help="Path to pipeline_config.yml",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = validate_configs(
        forecasting_config_path=Path(args.forecasting_config),
        pipeline_config_path=Path(args.pipeline_config),
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("=== Forecasting Config Integrity ===")
        print(f"ok: {report['ok']}")
        print(f"errors: {len(report['errors'])}")
        for err in report["errors"]:
            print(f"  - {err}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
