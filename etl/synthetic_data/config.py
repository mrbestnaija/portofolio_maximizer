from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merged[k] = _deep_merge(base[k], v)
        else:
            merged[k] = v
    return merged


def load_base_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    try:
        return yaml.safe_load(config_path.read_text()) or {}
    except Exception:
        return {}


def load_profile(config_root: Path, profile_name: Optional[str], profiles_path: Optional[Path]) -> Dict[str, Any]:
    if not profile_name:
        return {}
    profile_file = profiles_path or (config_root.parent / "synthetic_data_profiles.yml")
    if not profile_file.exists():
        return {}
    try:
        profiles = yaml.safe_load(profile_file.read_text()) or {}
    except Exception:
        return {}
    profile_block = profiles.get("profiles", {}).get(profile_name, {})
    return profile_block if isinstance(profile_block, dict) else {}


def load_synthetic_config(
    config_path: Path,
    profiles_path: Optional[Path] = None,
    profile_name: Optional[str] = None,
) -> Dict[str, Any]:
    base_cfg = load_base_config(config_path)
    synthetic_root = base_cfg.get("synthetic", {}) if isinstance(base_cfg, dict) else {}
    profile_env = os.getenv("SYNTHETIC_PROFILE")
    profile = profile_env or profile_name or synthetic_root.get("profile")
    profile_overrides = load_profile(config_path, profile, profiles_path)
    if profile_overrides:
        merged = _deep_merge(synthetic_root, profile_overrides)
        return {"synthetic": merged}
    return base_cfg or {}
