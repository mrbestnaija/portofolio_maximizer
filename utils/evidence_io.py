from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional


ValidationResult = Callable[[dict[str, Any]], bool | tuple[bool, str]]


def sha256_file(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}_",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(tmp_path, path)
        return path
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    indent: int = 2,
    sort_keys: bool = False,
    encoding: str = "utf-8",
    default: Optional[Callable[[Any], Any]] = None,
) -> Path:
    text = json.dumps(payload, indent=indent, sort_keys=sort_keys, default=default)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}_",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        with tmp_path.open("r", encoding=encoding) as handle:
            json.load(handle)
        os.replace(tmp_path, path)
        return path
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]], *, encoding: str = "utf-8") -> Path:
    lines = [json.dumps(record, separators=(",", ":")) for record in records]
    payload = "\n".join(lines)
    if payload:
        payload += "\n"
    return atomic_write_text(path, payload, encoding=encoding)


def load_json_file(path: Path, *, encoding: str = "utf-8") -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding=encoding))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_manifest_entry(
    artifact_path: Path,
    *,
    source: str,
    extra: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    digest = sha256_file(artifact_path)
    if not digest:
        return None
    try:
        size_bytes = int(artifact_path.stat().st_size)
    except Exception:
        size_bytes = None
    payload: dict[str, Any] = {
        "file": artifact_path.name,
        "sha256": digest,
        "bytes": size_bytes,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
    }
    if extra:
        payload.update({k: v for k, v in extra.items() if v is not None})
    return payload


def upsert_jsonl_record(
    path: Path,
    record: dict[str, Any],
    *,
    key_field: str = "file",
    encoding: str = "utf-8",
) -> dict[str, Any]:
    existing: list[dict[str, Any]] = []
    record_key = record.get(key_field)
    if path.exists():
        for raw_line in path.read_text(encoding=encoding, errors="replace").splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                parsed = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            if record_key is not None and parsed.get(key_field) == record_key:
                if parsed.get("sha256") == record.get("sha256"):
                    return {"updated": False, "record": parsed}
                continue
            existing.append(parsed)
    existing.append(record)
    atomic_write_jsonl(path, existing, encoding=encoding)
    return {"updated": True, "record": record}


def quarantine_file(
    path: Path,
    *,
    quarantine_dir: Path,
    reason: str,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target = quarantine_dir / f"{path.stem}_{stamp}{path.suffix}"
    os.replace(path, target)
    sidecar = target.with_suffix(target.suffix + ".meta.json")
    meta_payload = {
        "reason": reason,
        "original_path": str(path),
        "quarantined_path": str(target),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        meta_payload["metadata"] = metadata
    atomic_write_json(sidecar, meta_payload)
    return {"path": str(target), "meta_path": str(sidecar), "reason": reason}


def write_promoted_json_artifact(
    *,
    stamped_path: Path,
    latest_path: Optional[Path],
    payload: dict[str, Any],
    validate_fn: Optional[ValidationResult] = None,
    quarantine_dir: Optional[Path] = None,
) -> dict[str, Any]:
    atomic_write_json(stamped_path, payload)
    parsed = load_json_file(stamped_path)
    validation_reason = "ok"
    valid = bool(parsed)
    if validate_fn is not None:
        validation = validate_fn(parsed)
        if isinstance(validation, tuple):
            valid, validation_reason = bool(validation[0]), str(validation[1] or "invalid")
        else:
            valid = bool(validation)
            validation_reason = "ok" if valid else "invalid"
    if not valid:
        result: dict[str, Any] = {
            "ok": False,
            "stamped_path": str(stamped_path),
            "validation_reason": validation_reason,
        }
        if quarantine_dir is not None and stamped_path.exists():
            result["quarantine"] = quarantine_file(
                stamped_path,
                quarantine_dir=quarantine_dir,
                reason=validation_reason,
                metadata={"latest_path": str(latest_path) if latest_path else None},
            )
        return result
    if latest_path is not None:
        atomic_write_json(latest_path, parsed)
    return {
        "ok": True,
        "stamped_path": str(stamped_path),
        "latest_path": str(latest_path) if latest_path is not None else None,
        "validation_reason": validation_reason,
    }


def write_versioned_json_artifact(
    *,
    latest_path: Path,
    payload: dict[str, Any],
    archive_root: Optional[Path] = None,
    archive_name: Optional[str] = None,
    validate_fn: Optional[ValidationResult] = None,
    quarantine_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Write a latest JSON artifact plus an immutable timestamped archive copy.

    The latest path remains the operator-facing pointer, while the archive copy
    preserves a historical record for later audits, calibration review, and PnL
    provenance checks.
    """
    base_name = (archive_name or latest_path.stem or "artifact").strip() or "artifact"
    if archive_root is None:
        archive_root = latest_path.parent / f"{base_name}_history"

    stamped_path = archive_root / f"{base_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}{latest_path.suffix}"
    result = write_promoted_json_artifact(
        stamped_path=stamped_path,
        latest_path=latest_path,
        payload=payload,
        validate_fn=validate_fn,
        quarantine_dir=quarantine_dir,
    )
    result["archive_root"] = str(archive_root)
    result["archive_path"] = str(stamped_path)

    if result.get("ok"):
        manifest_path = archive_root / "manifest.jsonl"
        manifest_entry = build_manifest_entry(
            stamped_path,
            source="utils.evidence_io.write_versioned_json_artifact",
            extra={"latest_path": str(latest_path)},
        )
        if manifest_entry is not None:
            upsert_jsonl_record(manifest_path, manifest_entry)
            result["manifest_path"] = str(manifest_path)

    return result
