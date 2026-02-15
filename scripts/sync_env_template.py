"""Synchronize .env and .env.template (keys only, never values).

Safety:
- This script NEVER copies values from .env into .env.template.
- It only appends missing KEY=... placeholder lines.

Usage:
  python scripts/sync_env_template.py --dry-run
  python scripts/sync_env_template.py --write
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


_ASSIGN_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


def _extract_keys(path: Path) -> list[str]:
    keys: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _ASSIGN_RE.match(raw)
        if m:
            keys.append(m.group(1))
    return keys


def _placeholder_for(key: str) -> str:
    upper = key.upper()
    # Docker secrets convention.
    if upper.endswith("_FILE"):
        base = key[:-5].lower()
        return f"/run/secrets/{base}"
    # Secrets first: some keys contain both EMAIL and PASSWORD (e.g., OPENAI_EMAIL_PASSWORD).
    if any(tok in upper for tok in ("PASSWORD", "SECRET", "TOKEN", "KEY")):
        return f"your_{key.lower()}_here"
    if "URL" in upper:
        return "https://example.com"
    if "EMAIL" in upper:
        return "you@example.com"
    if "PHONE" in upper:
        return "+15551234567"
    if upper.endswith("_PORT"):
        return "8000"
    if upper in {"CI"}:
        return "false"
    return "your_value_here"


def _append_missing_keys(
    *,
    target_path: Path,
    missing_keys: list[str],
    header: str,
    placeholder: bool,
) -> None:
    if not missing_keys:
        return

    # If the target already contains this header (common when syncing iteratively),
    # insert the new keys into the existing block instead of creating duplicate blocks.
    try:
        existing_lines = target_path.read_text(encoding="utf-8", errors="replace").splitlines()
        marker = header.strip()
        marker_idxs = [i for i, line in enumerate(existing_lines) if line.strip() == marker]
        if marker_idxs:
            start_idx = marker_idxs[-1]
            insert_at = None
            for j in range(start_idx + 1, len(existing_lines)):
                if existing_lines[j].strip() == "":
                    insert_at = j
                    break
            if insert_at is None:
                insert_at = len(existing_lines)

            to_add = []
            for k in missing_keys:
                val = _placeholder_for(k) if placeholder else ""
                to_add.append(f"{k}={val}")

            merged = existing_lines[:insert_at] + to_add + existing_lines[insert_at:]
            target_path.write_text("\n".join(merged) + "\n", encoding="utf-8", newline="\n")
            return
    except Exception:
        pass

    lines: list[str] = []
    lines.append("")
    lines.append("# ============================================")
    lines.append(header)
    lines.append("# ============================================")
    if placeholder:
        lines.append("# Placeholders only. Never put real secrets in .env.template.")
    else:
        lines.append("# Added to keep key set synchronized; fill in as needed.")
    for k in missing_keys:
        val = _placeholder_for(k) if placeholder else ""
        lines.append(f"{k}={val}")

    # Append with a trailing newline for cleanliness.
    content = "\n".join(lines) + "\n"
    with target_path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(content)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=".env", help="Path to .env")
    parser.add_argument("--template", default=".env.template", help="Path to .env.template")
    parser.add_argument("--write", action="store_true", help="Apply changes (otherwise dry-run).")
    parser.add_argument("--dry-run", action="store_true", help="Print differences only (default).")
    args = parser.parse_args()

    if args.write and args.dry_run:
        raise SystemExit("Choose one: --write or --dry-run (default).")

    env_path = Path(args.env)
    tpl_path = Path(args.template)

    if not env_path.exists():
        raise SystemExit(f"Missing env file: {env_path}")
    if not tpl_path.exists():
        raise SystemExit(f"Missing template file: {tpl_path}")

    env_keys = set(_extract_keys(env_path))
    tpl_keys = set(_extract_keys(tpl_path))

    missing_in_tpl = sorted(env_keys - tpl_keys)
    missing_in_env = sorted(tpl_keys - env_keys)

    print(f".env unique keys: {len(env_keys)}")
    print(f".env.template unique keys: {len(tpl_keys)}")
    print(f"Missing in .env.template: {len(missing_in_tpl)}")
    print(f"Missing in .env: {len(missing_in_env)}")
    if missing_in_tpl:
        print("  Keys missing in template:")
        for k in missing_in_tpl:
            print(f"  - {k}")
    if missing_in_env:
        print("  Keys missing in env:")
        for k in missing_in_env:
            print(f"  - {k}")

    if not args.write:
        return 0

    _append_missing_keys(
        target_path=tpl_path,
        missing_keys=missing_in_tpl,
        header="# Additional variables (observed in .env) - synced by scripts/sync_env_template.py",
        placeholder=True,
    )
    _append_missing_keys(
        target_path=env_path,
        missing_keys=missing_in_env,
        header="# Added from .env.template - synced by scripts/sync_env_template.py",
        placeholder=False,
    )

    # Re-check.
    env_keys2 = set(_extract_keys(env_path))
    tpl_keys2 = set(_extract_keys(tpl_path))
    if env_keys2 != tpl_keys2:
        only_env = sorted(env_keys2 - tpl_keys2)
        only_tpl = sorted(tpl_keys2 - env_keys2)
        print("ERROR: Key sets still differ after write.")
        if only_env:
            print("  Only in .env:")
            for k in only_env:
                print(f"  - {k}")
        if only_tpl:
            print("  Only in .env.template:")
            for k in only_tpl:
                print(f"  - {k}")
        return 2

    print("OK: .env and .env.template now have identical key sets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
