from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


FALSEY_ENV_VALUES = {"0", "false", "no", "off"}
APPROVED_BENCHMARK_STATUSES = {"PASS", "OK", "APPROVED"}
DEFAULT_QWEN35_POLICY_PATH = Path("logs") / "openclaw_model_policy.json"


@dataclass(frozen=True)
class Qwen35Policy:
    fallback_allowed: bool
    primary_allowed: bool
    preferred_primary: Optional[str]
    policy_path: Optional[Path]
    benchmark_status: str
    benchmark_model: Optional[str]
    source: str


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _env_enabled(raw: Any, *, default: bool = False) -> bool:
    text = _coerce_text(raw).lower()
    if not text:
        return bool(default)
    return text not in FALSEY_ENV_VALUES


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = _coerce_text(value).lower()
    if not text:
        return None
    if text in FALSEY_ENV_VALUES:
        return False
    if text in {"1", "true", "yes", "on", "enabled", "enable", "approved", "pass"}:
        return True
    return None


def is_qwen35_variant(model_id: str) -> bool:
    return "qwen3.5" in _coerce_text(model_id).lower()


def _resolve_policy_path(
    env: Mapping[str, Any],
    *,
    base_dir: Optional[Path],
    policy_path: Optional[Path],
) -> Path:
    if policy_path is not None:
        candidate = Path(policy_path).expanduser()
    else:
        raw = _coerce_text(env.get("OPENCLAW_QWEN35_POLICY_PATH"))
        if raw:
            candidate = Path(raw).expanduser()
        else:
            candidate = DEFAULT_QWEN35_POLICY_PATH

    if candidate.is_absolute():
        return candidate

    base = Path(base_dir) if base_dir is not None else Path.cwd()
    return (base / candidate).resolve()


def _load_policy_payload(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        return {}, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}, "invalid"
    if not isinstance(payload, dict):
        return {}, "invalid"
    return payload, "present"


def _merge_policy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    merged = dict(payload)
    qwen35_section = payload.get("qwen35")
    if isinstance(qwen35_section, dict):
        merged.update(qwen35_section)
    benchmark_section = payload.get("benchmark")
    if isinstance(benchmark_section, dict):
        merged.update(benchmark_section)
    return merged


def _extract_text(*values: Any) -> str:
    for value in values:
        text = _coerce_text(value)
        if text:
            return text
    return ""


def load_qwen35_policy(
    *,
    env: Mapping[str, Any] | None = None,
    base_dir: Optional[Path] = None,
    policy_path: Optional[Path] = None,
) -> Qwen35Policy:
    env_map = dict(os.environ if env is None else env)
    resolved_path = _resolve_policy_path(env_map, base_dir=base_dir, policy_path=policy_path)

    env_fallback = _env_enabled(env_map.get("OPENCLAW_ALLOW_QWEN35_FALLBACK"), default=False)
    payload, path_state = _load_policy_payload(resolved_path)

    benchmark_status = "MISSING" if path_state == "missing" else ("INVALID" if path_state == "invalid" else "UNKNOWN")
    preferred_primary = ""
    benchmark_model = ""
    file_fallback_allowed = False
    file_primary_allowed = False

    if path_state == "present":
        merged = _merge_policy_payload(payload)
        benchmark_status = _extract_text(
            merged.get("status"),
            merged.get("benchmark_status"),
            payload.get("status"),
            payload.get("benchmark_status"),
            merged.get("result"),
            payload.get("result"),
            merged.get("overall_status"),
            payload.get("overall_status"),
        ).upper() or "UNKNOWN"
        preferred_primary = _extract_text(
            merged.get("preferred_primary"),
            merged.get("preferred_primary_model"),
            merged.get("approved_primary_model"),
            merged.get("model"),
            payload.get("preferred_primary"),
            payload.get("preferred_primary_model"),
            payload.get("approved_primary_model"),
            payload.get("model"),
        )
        benchmark_model = _extract_text(
            merged.get("benchmark_model"),
            payload.get("benchmark_model"),
            preferred_primary,
        )
        file_fallback_allowed = bool(
            _coerce_bool(merged.get("fallback_allowed"))
            or _coerce_bool(merged.get("allow_fallback"))
            or _coerce_bool(payload.get("fallback_allowed"))
            or _coerce_bool(payload.get("allow_fallback"))
        )
        file_primary_allowed = bool(
            _coerce_bool(merged.get("primary_allowed"))
            or _coerce_bool(merged.get("allow_primary"))
            or _coerce_bool(payload.get("primary_allowed"))
            or _coerce_bool(payload.get("allow_primary"))
        )

    benchmark_approved = benchmark_status in APPROVED_BENCHMARK_STATUSES
    primary_allowed = bool(
        benchmark_approved and file_primary_allowed and is_qwen35_variant(preferred_primary)
    )
    fallback_allowed = bool(
        env_fallback
        or (benchmark_approved and (file_fallback_allowed or primary_allowed))
    )

    source_parts: list[str] = []
    if env_fallback:
        source_parts.append("env")
    if path_state == "present":
        source_parts.append("file")
    elif path_state == "invalid":
        source_parts.append("file-invalid")
    elif path_state == "missing" and not source_parts:
        source_parts.append("default")

    return Qwen35Policy(
        fallback_allowed=fallback_allowed,
        primary_allowed=primary_allowed,
        preferred_primary=preferred_primary or None,
        policy_path=resolved_path,
        benchmark_status=benchmark_status,
        benchmark_model=benchmark_model or None,
        source="+".join(source_parts),
    )

