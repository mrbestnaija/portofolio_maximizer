#!/usr/bin/env python3
"""
OpenClaw TLS/DNS diagnostics for WhatsApp gateway reachability.

Checks:
- DNS resolution for each host.
- TLS handshake on the target port (default: 443).
- Proxy environment variables that can alter network behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import ssl
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dns_family(value: str) -> int:
    mode = str(value or "any").strip().lower()
    if mode == "ipv4":
        return socket.AF_INET
    if mode == "ipv6":
        return socket.AF_INET6
    return socket.AF_UNSPEC


def _proxy_env() -> dict[str, str]:
    out: dict[str, str] = {}
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY"):
        value = str(os.getenv(key, "")).strip()
        if value:
            out[key] = value
    return out


def _dns_probe(hostname: str, *, family: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "hostname": hostname,
        "ok": False,
        "addresses": [],
        "error": "",
        "latency_ms": 0,
    }
    started = time.perf_counter()
    try:
        infos = socket.getaddrinfo(str(hostname), None, family=family, type=socket.SOCK_STREAM)
        addresses = sorted(
            {
                str(row[4][0])
                for row in infos
                if isinstance(row, tuple)
                and len(row) >= 5
                and isinstance(row[4], tuple)
                and len(row[4]) >= 1
                and row[4][0]
            }
        )
        result["addresses"] = addresses[:12]
        result["ok"] = bool(addresses)
        if not addresses:
            result["error"] = "no_addresses"
    except Exception as exc:
        result["error"] = str(exc)
    result["latency_ms"] = int(round((time.perf_counter() - started) * 1000.0))
    return result


def _format_cert_name(raw: Any) -> str:
    parts: list[str] = []
    if not isinstance(raw, tuple):
        return ""
    for row in raw:
        if not isinstance(row, tuple):
            continue
        for item in row:
            if isinstance(item, tuple) and len(item) == 2 and item[0] and item[1]:
                parts.append(f"{item[0]}={item[1]}")
    return ", ".join(parts)


def _tls_probe_unverified(hostname: str, *, port: int, timeout_seconds: float) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ok": False,
        "peer_ip": "",
        "error": "",
        "connect_ms": 0,
        "handshake_ms": 0,
    }
    connect_start = time.perf_counter()
    try:
        with socket.create_connection((hostname, int(port)), timeout=max(1.0, float(timeout_seconds))) as tcp_sock:
            out["connect_ms"] = int(round((time.perf_counter() - connect_start) * 1000.0))
            try:
                out["peer_ip"] = str(tcp_sock.getpeername()[0])
            except Exception:
                out["peer_ip"] = ""

            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            handshake_start = time.perf_counter()
            with ctx.wrap_socket(tcp_sock, server_hostname=hostname):
                out["handshake_ms"] = int(round((time.perf_counter() - handshake_start) * 1000.0))
                out["ok"] = True
    except Exception as exc:
        out["error"] = str(exc)
        if out["connect_ms"] == 0:
            out["connect_ms"] = int(round((time.perf_counter() - connect_start) * 1000.0))
    return out


def _tls_probe(hostname: str, *, port: int, timeout_seconds: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "hostname": hostname,
        "port": int(port),
        "ok": False,
        "peer_ip": "",
        "connect_ms": 0,
        "handshake_ms": 0,
        "protocol": "",
        "cipher": "",
        "certificate": {},
        "verify_error": False,
        "error": "",
    }
    connect_start = time.perf_counter()
    try:
        with socket.create_connection((hostname, int(port)), timeout=max(1.0, float(timeout_seconds))) as tcp_sock:
            result["connect_ms"] = int(round((time.perf_counter() - connect_start) * 1000.0))
            try:
                result["peer_ip"] = str(tcp_sock.getpeername()[0])
            except Exception:
                result["peer_ip"] = ""

            ctx = ssl.create_default_context()
            handshake_start = time.perf_counter()
            with ctx.wrap_socket(tcp_sock, server_hostname=hostname) as tls_sock:
                result["handshake_ms"] = int(round((time.perf_counter() - handshake_start) * 1000.0))
                result["protocol"] = str(tls_sock.version() or "")
                cipher = tls_sock.cipher()
                if isinstance(cipher, tuple) and len(cipher) >= 1:
                    result["cipher"] = str(cipher[0])

                cert = tls_sock.getpeercert() or {}
                if isinstance(cert, dict):
                    result["certificate"] = {
                        "subject": _format_cert_name(cert.get("subject")),
                        "issuer": _format_cert_name(cert.get("issuer")),
                        "not_before": str(cert.get("notBefore") or ""),
                        "not_after": str(cert.get("notAfter") or ""),
                    }
                result["ok"] = True
    except ssl.SSLCertVerificationError as exc:
        result["verify_error"] = True
        result["error"] = str(exc)
        fallback = _tls_probe_unverified(hostname, port=port, timeout_seconds=timeout_seconds)
        result["unverified_reachable"] = bool(fallback.get("ok"))
        result["unverified_error"] = str(fallback.get("error") or "")
        if str(result.get("peer_ip") or "").strip() == "" and str(fallback.get("peer_ip") or "").strip():
            result["peer_ip"] = str(fallback.get("peer_ip"))
        if int(result.get("connect_ms") or 0) <= 0:
            result["connect_ms"] = int(fallback.get("connect_ms") or 0)
        if int(result.get("handshake_ms") or 0) <= 0:
            result["handshake_ms"] = int(fallback.get("handshake_ms") or 0)
    except Exception as exc:
        result["error"] = str(exc)
        if int(result.get("connect_ms") or 0) <= 0:
            result["connect_ms"] = int(round((time.perf_counter() - connect_start) * 1000.0))
    return result


def _recommendations(
    *,
    host: str,
    dns_result: dict[str, Any],
    tls_result: dict[str, Any],
    proxy_env: dict[str, str],
) -> list[str]:
    out: list[str] = []
    if not bool(dns_result.get("ok")):
        out.append(f"DNS failed for {host}. Verify resolver, firewall, and outbound network policy.")
        out.append("Run `nslookup web.whatsapp.com` and compare with this host's DNS settings.")
    if bool(tls_result.get("verify_error")):
        out.append(
            "TLS certificate verification failed. Check corporate TLS interception and configure trust chain (for Node: NODE_EXTRA_CA_CERTS)."
        )
    tls_error = str(tls_result.get("error") or "").lower()
    if "timed out" in tls_error:
        out.append("TLS handshake timed out. Validate outbound 443 access and proxy behavior.")
    if proxy_env:
        out.append("Proxy environment variables are set. Ensure they allow websocket/TLS traffic to WhatsApp hosts.")
    if not out:
        out.append("DNS and TLS checks are healthy for this host.")
    return out


def _diagnose_host(
    *,
    hostname: str,
    port: int,
    timeout_seconds: float,
    family: int,
    proxy_env: dict[str, str],
) -> dict[str, Any]:
    dns_result = _dns_probe(hostname, family=family)
    tls_result = _tls_probe(hostname, port=port, timeout_seconds=timeout_seconds)
    ok = bool(dns_result.get("ok")) and bool(tls_result.get("ok"))
    return {
        "hostname": hostname,
        "port": int(port),
        "status": "PASS" if ok else "FAIL",
        "dns": dns_result,
        "tls": tls_result,
        "recommendations": _recommendations(
            host=hostname,
            dns_result=dns_result,
            tls_result=tls_result,
            proxy_env=proxy_env,
        ),
    }


def _print_human(report: dict[str, Any]) -> None:
    print("[openclaw_tls_dns_diagnostics] Connectivity report")
    print(f"timestamp_utc: {report.get('timestamp_utc')}")
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    print(
        "summary: "
        f"hosts={summary.get('hosts_total', 0)} pass={summary.get('pass_count', 0)} fail={summary.get('fail_count', 0)}"
    )
    proxy_env = report.get("proxy_environment") if isinstance(report.get("proxy_environment"), dict) else {}
    if proxy_env:
        print("proxy_environment:")
        for key in sorted(proxy_env.keys()):
            print(f"  {key}={proxy_env[key]}")
    hosts = report.get("hosts") if isinstance(report.get("hosts"), list) else []
    for row in hosts:
        if not isinstance(row, dict):
            continue
        dns_row = row.get("dns") if isinstance(row.get("dns"), dict) else {}
        tls_row = row.get("tls") if isinstance(row.get("tls"), dict) else {}
        print("")
        print(f"{row.get('hostname')}:{row.get('port')} status={row.get('status')}")
        if bool(dns_row.get("ok")):
            addresses = dns_row.get("addresses") if isinstance(dns_row.get("addresses"), list) else []
            print(f"  dns: ok latency_ms={dns_row.get('latency_ms')} addresses={','.join(str(x) for x in addresses)}")
        else:
            print(f"  dns: fail error={dns_row.get('error')}")
        if bool(tls_row.get("ok")):
            print(
                "  tls: ok "
                f"peer_ip={tls_row.get('peer_ip')} protocol={tls_row.get('protocol')} cipher={tls_row.get('cipher')} "
                f"connect_ms={tls_row.get('connect_ms')} handshake_ms={tls_row.get('handshake_ms')}"
            )
        else:
            print(
                "  tls: fail "
                f"peer_ip={tls_row.get('peer_ip')} error={tls_row.get('error')} "
                f"verify_error={tls_row.get('verify_error')}"
            )
        recs = row.get("recommendations") if isinstance(row.get("recommendations"), list) else []
        for rec in recs:
            print(f"  advice: {rec}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        action="append",
        dest="hosts",
        help="Host to test. Can be repeated. Defaults to WhatsApp hosts if omitted.",
    )
    parser.add_argument("--port", type=int, default=443, help="TLS target port (default: 443).")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=8.0,
        help="Socket timeout for TLS checks (default: 8).",
    )
    parser.add_argument(
        "--family",
        choices=("any", "ipv4", "ipv6"),
        default="any",
        help="DNS address family preference.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    parser.add_argument("--output-file", default="", help="Optional path to persist the JSON report.")
    args = parser.parse_args(argv)

    hosts = [str(x).strip() for x in (args.hosts or []) if str(x).strip()]
    if not hosts:
        hosts = ["web.whatsapp.com", "mmg.whatsapp.net"]

    family = _dns_family(args.family)
    proxy_environment = _proxy_env()
    host_reports = [
        _diagnose_host(
            hostname=host,
            port=max(1, int(args.port)),
            timeout_seconds=max(1.0, float(args.timeout_seconds)),
            family=family,
            proxy_env=proxy_environment,
        )
        for host in hosts
    ]

    pass_count = sum(1 for row in host_reports if str(row.get("status")) == "PASS")
    fail_count = len(host_reports) - pass_count
    report: dict[str, Any] = {
        "timestamp_utc": _utc_now_iso(),
        "proxy_environment": proxy_environment,
        "hosts": host_reports,
        "summary": {
            "hosts_total": len(host_reports),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "all_ok": fail_count == 0,
        },
    }

    if str(args.output_file or "").strip():
        out_path = Path(str(args.output_file)).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_human(report)

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
