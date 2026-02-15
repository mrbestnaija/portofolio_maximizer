#!/usr/bin/env python3
"""
Windows TTS -> WAV -> OpenClaw media notification.

This adds a practical "audio ability" without introducing new Python deps:
- Generates a WAV using Windows `System.Speech`
- Sends it via OpenClaw (`openclaw message send --media <wav>`)

Notes:
- Requires Windows PowerShell and the .NET `System.Speech` assembly.
- Works best with Telegram/Discord; WhatsApp can deliver the media but you may
  not get a phone push notification for "messages you sent to yourself".
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _read_stdin() -> str:
    try:
        if sys.stdin is None or sys.stdin.closed or sys.stdin.isatty():
            return ""
        return sys.stdin.read()
    except Exception:
        return ""


def _redact_text(text: str) -> str:
    payload = text or ""
    secret_markers = ("KEY", "TOKEN", "SECRET", "PASSWORD")
    for name, value in os.environ.items():
        if not value or len(value) < 8:
            continue
        upper = name.upper()
        if any(marker in upper for marker in secret_markers):
            payload = payload.replace(value, "[REDACTED]")
    return payload


def _generate_wav_windows(*, text: str, out_path: Path, voice: Optional[str] = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PMX_TTS_TEXT"] = text
    env["PMX_TTS_OUT"] = str(out_path)
    if voice:
        env["PMX_TTS_VOICE"] = voice

    # Keep PowerShell script tiny; take inputs from env to avoid quoting traps.
    ps = r"""
$out = $env:PMX_TTS_OUT
$text = $env:PMX_TTS_TEXT
$voice = $env:PMX_TTS_VOICE
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
if ($voice -and $voice.Trim().Length -gt 0) {
  try { $synth.SelectVoice($voice) } catch { }
}
$synth.SetOutputToWaveFile($out)
$synth.Speak($text)
$synth.Dispose()
"""

    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout or "").splitlines()[-20:])
        raise RuntimeError(f"TTS generation failed (exit={proc.returncode}). Tail:\n{tail}")

    if not out_path.exists() or out_path.stat().st_size <= 0:
        raise RuntimeError("TTS generation produced no output file.")


def _resolve_targets_from_args(args) -> list[tuple[Optional[str], str]]:
    default_channel = (os.getenv("OPENCLAW_CHANNEL") or "").strip() or None

    from utils.openclaw_cli import infer_linked_whatsapp_target, parse_openclaw_targets, resolve_openclaw_targets

    if (args.targets or "").strip():
        return parse_openclaw_targets((args.targets or "").strip(), default_channel=default_channel)
    if (args.to or "").strip():
        return parse_openclaw_targets((args.to or "").strip(), default_channel=default_channel)

    env_targets = (os.getenv("OPENCLAW_TARGETS") or "").strip()
    env_to = (os.getenv("OPENCLAW_TO") or "").strip()
    targets = resolve_openclaw_targets(env_targets=env_targets, env_to=env_to, cfg_to=None, default_channel=default_channel)
    if targets:
        return targets

    if bool(args.infer_to):
        inferred = infer_linked_whatsapp_target(command=str(args.command or "openclaw"), cwd=PROJECT_ROOT, timeout_seconds=10.0)
        if inferred:
            return [(None, inferred)]

    return []


def main(argv: list[str]) -> int:
    # Load `.env` safely (best-effort) without printing or overwriting existing env vars.
    try:
        from etl.secret_loader import bootstrap_dotenv

        bootstrap_dotenv()
    except Exception:
        pass

    p = argparse.ArgumentParser(description="Generate TTS audio (WAV) and send via OpenClaw as media.")
    p.add_argument("--text", default="", help="Text to speak. If omitted, reads from stdin.")
    p.add_argument("--voice", default="", help="Optional Windows voice name (best-effort).")
    p.add_argument("--out", default="", help="Output WAV path (default: logs/tts/tts_<ts>.wav).")
    p.add_argument("--notify-text", default="", help="Optional caption/message to send alongside the media.")

    p.add_argument("--to", nargs="?", default="", help="OpenClaw target or comma-separated targets (optional).")
    p.add_argument("--targets", default="", help="OpenClaw multi-target string (optional).")
    p.add_argument("--infer-to", dest="infer_to", action="store_true", default=True, help="Infer WhatsApp self target if none configured.")
    p.add_argument("--no-infer-to", dest="infer_to", action="store_false", help="Disable inference fallback.")
    p.add_argument("--command", default=os.getenv("OPENCLAW_COMMAND", "openclaw"), help='OpenClaw command (default: "openclaw").')
    p.add_argument("--timeout-seconds", type=float, default=float(os.getenv("OPENCLAW_TIMEOUT_SECONDS", "20") or 20), help="OpenClaw send timeout.")

    args = p.parse_args(argv)

    text = (args.text or "").strip() or _read_stdin().strip()
    if not text:
        print("[pmx_tts_notify] Missing --text (and stdin was empty).", file=sys.stderr)
        return 2

    if platform.system().lower() != "windows":
        print("[pmx_tts_notify] This helper currently supports Windows only.", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = (args.out or "").strip()
    if out:
        out_path = Path(out).expanduser().resolve()
    else:
        out_path = (PROJECT_ROOT / "logs" / "tts" / f"tts_{stamp}.wav").resolve()

    voice = (args.voice or "").strip() or None
    try:
        _generate_wav_windows(text=text, out_path=out_path, voice=voice)
    except Exception as exc:
        print(f"[pmx_tts_notify] FAILED generating WAV: {exc}", file=sys.stderr)
        return 1

    targets = _resolve_targets_from_args(args)
    if not targets:
        print("[pmx_tts_notify] No OpenClaw targets configured (set OPENCLAW_TARGETS/OPENCLAW_TO or pass --targets/--to).", file=sys.stderr)
        print(f"[pmx_tts_notify] WAV saved at: {out_path}", file=sys.stderr)
        return 2

    caption = (args.notify_text or "").strip()
    if not caption:
        caption = f"TTS audio ({out_path.name})"

    try:
        from utils.openclaw_cli import send_message_multi

        results = send_message_multi(
            targets=targets,
            message=_redact_text(caption),
            media=str(out_path),
            command=str(args.command or "openclaw"),
            cwd=PROJECT_ROOT,
            timeout_seconds=float(args.timeout_seconds),
        )
        if all(r.ok for r in results):
            print("[pmx_tts_notify] OK")
            return 0
        bad = next((r for r in results if not r.ok), results[0])
        print(f"[pmx_tts_notify] FAILED (exit={bad.returncode})", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[pmx_tts_notify] FAILED sending OpenClaw media: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

