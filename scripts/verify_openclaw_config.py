#!/usr/bin/env python3
"""
OpenClaw configuration verifier.

Validates ~/.openclaw/openclaw.json against the documented schema
(https://docs.openclaw.ai/gateway/configuration-reference.md) and
checks alignment with project config (config/llm_config.yml, .env).

Usage:
    python scripts/verify_openclaw_config.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.openclaw_cron_contract import load_cron_jobs_payload, summarize_cron_jobs

OPENCLAW_JSON = Path.home() / ".openclaw" / "openclaw.json"
OPENCLAW_CRON_JOBS = Path.home() / ".openclaw" / "cron" / "jobs.json"
RECOMMENDED_BOOTSTRAP_MAX_CHARS = 20000
WORKSPACE_BOOTSTRAP_FILES = ("SOUL.md", "AGENTS.md", "TOOLS.md", "IDENTITY.md", "USER.md")
FALSEY_VALUES = {"0", "false", "no", "off"}
VALID_OLLAMA_APIS = {
    "ollama",
    "openai-completions",
    "openai-responses",
    "anthropic-messages",
    "google-generative-ai",
}
VALID_EXEC_HOSTS: frozenset = frozenset({"sandbox", "gateway", "node"})
VALID_SANDBOX_MODES_FOR_SANDBOX_HOST: frozenset = frozenset({"non-main", "all"})


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _as_list(value):
    return value if isinstance(value, list) else []


def _agent_allows_exec(agent: dict) -> bool:
    tools = _as_dict(agent.get("tools"))
    deny = {str(item).strip().lower() for item in _as_list(tools.get("deny")) if str(item).strip()}
    if "exec" in deny or "group:runtime" in deny:
        return False

    allow = {str(item).strip().lower() for item in _as_list(tools.get("allow")) if str(item).strip()}
    if allow:
        return "exec" in allow or "group:runtime" in allow

    profile = str(tools.get("profile") or "").strip().lower()
    return profile != "messaging"


def _docker_sandbox_available(timeout_seconds: float = 5.0) -> bool:
    try:
        proc = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False
    return int(proc.returncode) == 0


def _env_enabled(raw: str | None, *, default: bool) -> bool:
    text = str(raw or "").strip().lower()
    if not text:
        return bool(default)
    return text not in FALSEY_VALUES


def _load_cfg() -> dict:
    return json.loads(OPENCLAW_JSON.read_text(encoding="utf-8-sig"))


def _load_cron_jobs_payload() -> tuple[dict, str | None]:
    return load_cron_jobs_payload(OPENCLAW_CRON_JOBS)


def _describe_agent_tools_policy(agent_tools: dict) -> str:
    profile = str(agent_tools.get("profile") or "").strip()
    if profile:
        return f"tools.profile={profile}"
    if _as_list(agent_tools.get("allow")) or _as_list(agent_tools.get("deny")):
        return "tools.policy=explicit"
    return "tools.profile=<inherit>"


def _load_env() -> dict:
    env_path = PROJECT_ROOT / ".env"
    env = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip().strip("'\"")
    return env


def _load_llm_config() -> dict:
    try:
        import yaml
        with open(PROJECT_ROOT / "config" / "llm_config.yml") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def main() -> int:
    issues: list[str] = []
    warnings: list[str] = []
    ok: list[str] = []

    cfg = _load_cfg()
    env = _load_env()
    llm_cfg = _load_llm_config()

    # ===== 1. MODEL PROVIDERS =====
    providers = cfg.get("models", {}).get("providers", {})

    # Ollama provider
    ollama = providers.get("ollama", {})
    if not ollama:
        issues.append("[CRITICAL] models.providers.ollama missing")
    else:
        if not ollama.get("baseUrl"):
            issues.append("[ERROR] ollama.baseUrl missing")
        else:
            ok.append(f"ollama.baseUrl = {ollama['baseUrl']}")

        api = ollama.get("api", "")
        if api not in VALID_OLLAMA_APIS:
            issues.append(f"[ERROR] ollama.api={api!r} not in {VALID_OLLAMA_APIS}")
        else:
            ok.append(f"ollama.api = {api}")
            base_url = str(ollama.get("baseUrl", "")).strip().lower()
            if api == "ollama" and base_url.endswith("/v1"):
                warnings.append(
                    "ollama.api='ollama' with baseUrl ending in '/v1' looks like an OpenAI-compatible endpoint; "
                    "prefer baseUrl without '/v1' for native Ollama API mode."
                )

        models = ollama.get("models", [])
        if not models:
            issues.append("[ERROR] ollama.models[] empty")
        else:
            ok.append(f"ollama.models count = {len(models)}")
            required_fields = ["id", "name", "reasoning", "input", "cost", "contextWindow", "maxTokens"]
            for m in models:
                mid = m.get("id", "?")
                missing = [k for k in required_fields if k not in m]
                if missing:
                    issues.append(f"[ERROR] model {mid} missing: {missing}")
                else:
                    ok.append(f"  {mid}: ctx={m['contextWindow']}, maxTok={m['maxTokens']}, reasoning={m['reasoning']}")

    # No remote providers
    remote_providers = [p for p in providers if p != "ollama"]
    if remote_providers:
        issues.append(f"[CRITICAL] Remote providers configured: {remote_providers}")
    else:
        ok.append("No remote/cloud providers (429-safe)")

    # ===== 2. AGENT DEFAULTS =====
    defaults = cfg.get("agents", {}).get("defaults", {})

    primary = defaults.get("model", {}).get("primary", "")
    fallbacks = defaults.get("model", {}).get("fallbacks", [])

    if not primary:
        issues.append("[CRITICAL] agents.defaults.model.primary not set")
    elif not primary.startswith("ollama/"):
        issues.append(f"[CRITICAL] Primary {primary} is remote (must be ollama/*)")
    else:
        ok.append(f"Primary model = {primary}")

    remote_fb = [fb for fb in fallbacks if not fb.startswith("ollama/")]
    if remote_fb:
        issues.append(f"[CRITICAL] Remote fallbacks: {remote_fb}")
    else:
        ok.append(f"Fallbacks = {fallbacks or '[] (none, local-only)'}")

    # Model allowlist
    allowlist = defaults.get("models", {})
    remote_allowed = [m for m in allowlist if not m.startswith("ollama/")]
    if remote_allowed:
        warnings.append(f"Remote models in allowlist: {remote_allowed}")
    else:
        ok.append(f"Allowlist ({len(allowlist)} entries): all local")

    # Image model
    img = defaults.get("imageModel", {})
    if img:
        ip = img.get("primary", "")
        if ip and not ip.startswith("ollama/"):
            warnings.append(f"imageModel.primary = {ip} (remote, may fail)")
        elif ip:
            ok.append(f"imageModel.primary = {ip}")
    else:
        ok.append("imageModel: not set (no remote dependency)")

    # Context management
    ctx_tokens = defaults.get("contextTokens", 200000)
    bootstrap_max = defaults.get("bootstrapMaxChars", RECOMMENDED_BOOTSTRAP_MAX_CHARS)
    ok.append(f"contextTokens = {ctx_tokens}")
    ok.append(f"bootstrapMaxChars = {bootstrap_max}")
    try:
        bootstrap_limit_int = int(bootstrap_max)
    except Exception:
        bootstrap_limit_int = RECOMMENDED_BOOTSTRAP_MAX_CHARS
    if ctx_tokens > 65536:
        warnings.append(f"contextTokens ({ctx_tokens}) very high for local models")
    if bootstrap_limit_int < RECOMMENDED_BOOTSTRAP_MAX_CHARS:
        warnings.append(
            "bootstrapMaxChars is below 20000; large workspace bootstrap files may be truncated "
            "(can cause malformed tool calls/stuck sessions)."
        )
    for fname in WORKSPACE_BOOTSTRAP_FILES:
        fpath = PROJECT_ROOT / fname
        if not fpath.exists():
            continue
        try:
            size = len(fpath.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if size > bootstrap_limit_int:
            warnings.append(
                f"{fname} is {size} chars > bootstrapMaxChars ({bootstrap_limit_int}); "
                "OpenClaw will truncate bootstrap context."
            )
        else:
            ok.append(f"{fname} size = {size} chars (within bootstrapMaxChars)")

    # Compaction
    compaction = defaults.get("compaction", {}).get("mode", "default")
    ok.append(f"compaction.mode = {compaction}")
    if compaction != "safeguard":
        warnings.append("compaction.mode should be 'safeguard' for small-context models")

    # ===== 3. GATEWAY =====
    gw = cfg.get("gateway", {})
    ok.append(f"gateway.mode = {gw.get('mode', 'local')}")
    ok.append(f"gateway.port = {gw.get('port', 18789)}")
    ok.append(f"gateway.bind = {gw.get('bind', 'loopback')}")
    auth_mode = gw.get("auth", {}).get("mode", "token")
    has_token = bool(gw.get("auth", {}).get("token"))
    ok.append(f"gateway.auth.mode = {auth_mode}")
    if auth_mode == "token" and not has_token:
        warnings.append("gateway.auth.token not set")
    else:
        ok.append("gateway.auth.token = [set]")

    # ===== 4. AUTH PROFILES =====
    auth_profiles = cfg.get("auth", {}).get("profiles", {})
    remote_auth = [k for k in auth_profiles if "qwen-portal" in k or "openai" in k or "anthropic" in k]
    if remote_auth:
        warnings.append(f"Remote auth profiles: {remote_auth}")
    else:
        ok.append(f"Auth profiles: clean ({len(auth_profiles)} entries)")

    # ===== 5. PLUGINS =====
    plugins = cfg.get("plugins", {}).get("entries", {})
    if plugins.get("qwen-portal-auth", {}).get("enabled", False):
        warnings.append("qwen-portal-auth plugin still enabled")
    else:
        ok.append("qwen-portal-auth plugin: disabled")

    # ===== 6. WHATSAPP CHANNEL =====
    wa = cfg.get("channels", {}).get("whatsapp", {})
    if wa:
        ok.append(f"whatsapp.dmPolicy = {wa.get('dmPolicy', 'pairing')}")
        ok.append(f"whatsapp.groupPolicy = {wa.get('groupPolicy', 'allowlist')}")
        ok.append(f"whatsapp.allowFrom = {len(wa.get('allowFrom', []))} numbers")
        ok.append(f"whatsapp.selfChatMode = {wa.get('selfChatMode', False)}")
        ok.append(f"whatsapp.debounceMs = {wa.get('debounceMs', 1000)}")
        ok.append(f"whatsapp.sendReadReceipts = {wa.get('sendReadReceipts', True)}")

        # Per docs, accounts.default.enabled must be true
        accts = wa.get("accounts", {})
        default_acct = accts.get("default", {})
        if not default_acct.get("enabled", False):
            warnings.append("whatsapp.accounts.default.enabled is false")
        else:
            ok.append("whatsapp.accounts.default.enabled = true")
    else:
        warnings.append("No WhatsApp channel configured")

    # ===== 7. SESSION =====
    session = cfg.get("session", {})
    ok.append(f"session.dmScope = {session.get('dmScope', 'main')}")

    # ===== 8. ENV ALIGNMENT =====
    env_local_only = env.get("OPENCLAW_LOCAL_ONLY", "")
    if env_local_only in {"1", "true", "yes", "on"}:
        ok.append(f".env OPENCLAW_LOCAL_ONLY = {env_local_only}")
    elif not env_local_only:
        warnings.append(".env missing OPENCLAW_LOCAL_ONLY (default is 1 in code)")
    else:
        warnings.append(f".env OPENCLAW_LOCAL_ONLY = {env_local_only} (remote models allowed!)")

    env_edge_safe_runtime = env.get("PMX_EDGE_SAFE_RUNTIME", "")
    edge_safe_enabled = bool(env_edge_safe_runtime) and _env_enabled(env_edge_safe_runtime, default=False)
    if edge_safe_enabled:
        ok.append(f".env PMX_EDGE_SAFE_RUNTIME = {env_edge_safe_runtime}")
        if not _env_enabled(env_local_only, default=False):
            issues.append("[ERROR] PMX_EDGE_SAFE_RUNTIME requires OPENCLAW_LOCAL_ONLY=1")
        for runtime_flag in ("ENABLE_PARALLEL_FORECASTS", "ENABLE_PARALLEL_TICKER_PROCESSING", "ENABLE_GPU_PARALLEL"):
            if _env_enabled(env.get(runtime_flag), default=False):
                issues.append(f"[ERROR] {runtime_flag} must be disabled when PMX_EDGE_SAFE_RUNTIME=1")
    elif env_edge_safe_runtime:
        warnings.append(f".env PMX_EDGE_SAFE_RUNTIME = {env_edge_safe_runtime} (disabled)")

    env_model_order = env.get("OPENCLAW_OLLAMA_MODEL_ORDER", "")
    if env_model_order:
        models_ordered = [m.strip() for m in env_model_order.split(",") if m.strip()]
        if models_ordered and "qwen3" not in models_ordered[0]:
            issues.append(f"[ERROR] OPENCLAW_OLLAMA_MODEL_ORDER first model is {models_ordered[0]}, not qwen3:8b (tool-calling required)")
        else:
            ok.append(f".env model order: {env_model_order}")

    env_autonomy_guard = env.get("OPENCLAW_AUTONOMY_GUARD_ENABLED", "")
    if env_autonomy_guard and not _env_enabled(env_autonomy_guard, default=True):
        issues.append("[CRITICAL] OPENCLAW_AUTONOMY_GUARD_ENABLED is disabled")
    elif env_autonomy_guard:
        ok.append(f".env OPENCLAW_AUTONOMY_GUARD_ENABLED = {env_autonomy_guard}")
    else:
        ok.append(".env OPENCLAW_AUTONOMY_GUARD_ENABLED not set (runtime default: enabled)")

    env_autonomy_approval = env.get("OPENCLAW_AUTONOMY_REQUIRE_APPROVAL_TOKEN", "")
    if env_autonomy_approval and not _env_enabled(env_autonomy_approval, default=True):
        issues.append("[CRITICAL] OPENCLAW_AUTONOMY_REQUIRE_APPROVAL_TOKEN is disabled")
    elif env_autonomy_approval:
        ok.append(f".env OPENCLAW_AUTONOMY_REQUIRE_APPROVAL_TOKEN = {env_autonomy_approval}")
    else:
        ok.append(".env OPENCLAW_AUTONOMY_REQUIRE_APPROVAL_TOKEN not set (runtime default: enabled)")

    env_autonomy_prefix = env.get("OPENCLAW_AUTONOMY_POLICY_PREFIX_ENABLED", "")
    if env_autonomy_prefix and not _env_enabled(env_autonomy_prefix, default=True):
        warnings.append("OPENCLAW_AUTONOMY_POLICY_PREFIX_ENABLED is disabled")
    elif env_autonomy_prefix:
        ok.append(f".env OPENCLAW_AUTONOMY_POLICY_PREFIX_ENABLED = {env_autonomy_prefix}")

    env_injection_block = env.get("OPENCLAW_AUTONOMY_BLOCK_INJECTION_PATTERNS", "")
    if env_injection_block and _env_enabled(env_injection_block, default=False):
        ok.append(f".env OPENCLAW_AUTONOMY_BLOCK_INJECTION_PATTERNS = {env_injection_block}")
    else:
        warnings.append(
            "OPENCLAW_AUTONOMY_BLOCK_INJECTION_PATTERNS is not enabled "
            "(recommended for fully autonomous web-heavy workflows)"
        )

    env_approval_token = env.get("OPENCLAW_AUTONOMY_APPROVAL_TOKEN", "")
    if env_approval_token:
        if len(env_approval_token.strip()) < 8:
            warnings.append("OPENCLAW_AUTONOMY_APPROVAL_TOKEN is short; use a non-trivial token")
        else:
            ok.append(".env OPENCLAW_AUTONOMY_APPROVAL_TOKEN = [set]")

    # ===== 9. LLM CONFIG ALIGNMENT =====
    llm = llm_cfg.get("llm", {})
    active_model = llm.get("active_model", "")
    primary_model = llm.get("models", {}).get("primary", {}).get("name", "")
    if active_model and "qwen3" not in active_model:
        warnings.append(f"llm_config.yml active_model = {active_model} (should be qwen3:8b)")
    elif active_model:
        ok.append(f"llm_config.yml active_model = {active_model}")
    if primary_model and "qwen3" not in primary_model:
        warnings.append(f"llm_config.yml primary model = {primary_model} (should be qwen3:8b)")
    elif primary_model:
        ok.append(f"llm_config.yml primary model = {primary_model}")

    # ===== 10. MULTI-AGENT ARCHITECTURE =====
    agent_list = cfg.get("agents", {}).get("list", [])
    bindings = cfg.get("bindings", [])
    tools_cfg = cfg.get("tools", {})
    exec_host = str(cfg.get("tools", {}).get("exec", {}).get("host", "") or "").strip().lower()

    if exec_host == "sandbox" and not _docker_sandbox_available():
        issues.append("[ERROR] Docker daemon unavailable for tools.exec.host='sandbox'; use gateway/node fallback on this host")

    if not agent_list:
        warnings.append("No agents.list defined (single-agent mode, may cause session contention)")
    else:
        ok.append(f"Multi-agent: {len(agent_list)} agents defined")

        # Check for exactly one default agent
        defaults_found = [a["id"] for a in agent_list if a.get("default")]
        if len(defaults_found) != 1:
            issues.append(f"[ERROR] Expected 1 default agent, found {len(defaults_found)}: {defaults_found}")
        else:
            ok.append(f"Default agent: {defaults_found[0]}")

        # Check each agent has required fields and unique agentDir
        agent_dirs = {}
        expected_agents = {"ops", "trading", "training", "notifier"}
        found_agents = set()
        for agent in agent_list:
            aid = agent.get("id", "?")
            found_agents.add(aid)

            # Unique agentDir (per docs: never reuse)
            adir = agent.get("agentDir", "")
            if adir in agent_dirs:
                issues.append(f"[CRITICAL] agentDir collision: {aid} and {agent_dirs[adir]} share {adir}")
            elif adir:
                agent_dirs[adir] = aid

            # Workspace must exist
            ws = agent.get("workspace", "")
            if ws and not Path(ws).exists():
                warnings.append(f"Agent {aid}: workspace does not exist: {ws}")
            elif ws:
                ok.append(f"  {aid}: workspace exists")

            # Model must be local
            model = agent.get("model", "")
            if model and not model.startswith("ollama/"):
                issues.append(f"[CRITICAL] Agent {aid} uses remote model: {model}")
            elif model:
                ok.append(f"  {aid}: model = {model}")

            # Tool sandboxing
            agent_tools = agent.get("tools", {})
            deny = agent_tools.get("deny", [])
            ok.append(f"  {aid}: {_describe_agent_tools_policy(agent_tools)}, deny={deny or '[]'}")

            if exec_host == "sandbox" and _agent_allows_exec(agent):
                agent_sandbox = _as_dict(agent.get("sandbox"))
                agent_mode = str(agent_sandbox.get("mode") or "").strip().lower()
                if agent_mode and agent_mode not in VALID_SANDBOX_MODES_FOR_SANDBOX_HOST:
                    issues.append(
                        f"[ERROR] Agent {aid} sandbox.mode={agent_mode!r} overrides sandbox host and disables exec sessions"
                    )
                else:
                    ok.append(f"  {aid}: sandbox.mode = {agent_mode or '<inherit>'}")

        missing_agents = expected_agents - found_agents
        if missing_agents:
            warnings.append(f"Expected agents not found: {missing_agents}")
        else:
            ok.append("All 4 expected agents present (ops, trading, training, notifier)")

    # Bindings validation
    if not bindings:
        if agent_list:
            warnings.append("No bindings defined (all messages go to default agent)")
    else:
        ok.append(f"Bindings: {len(bindings)} routing rules")
        bound_agents = set()
        for b in bindings:
            bid = b.get("agentId", "?")
            match = b.get("match", {})
            channel = match.get("channel", "?")
            account = match.get("accountId", "*")
            bound_agents.add(bid)
            ok.append(f"  {channel}/{account} -> {bid}")

        # Check bound agents exist in agent list
        defined_ids = {a.get("id") for a in agent_list} if agent_list else set()
        orphan_bindings = bound_agents - defined_ids
        if orphan_bindings:
            issues.append(f"[ERROR] Bindings reference undefined agents: {orphan_bindings}")

    # ===== 11. CRON JOB SCHEMA =====
    cron_payload, cron_error = _load_cron_jobs_payload()
    cron_summary = summarize_cron_jobs(cron_payload)
    if cron_error:
        issues.append(f"[CRITICAL] {cron_error}")
    elif cron_summary["status"] == "FAIL":
        invalid_session_targets = int(cron_summary.get("invalid_session_target_count", 0) or 0)
        malformed_jobs = int(cron_summary.get("jobs_invalid", 0) or 0)
        if invalid_session_targets > 0:
            issues.append(
                "[CRITICAL] Cron jobs contain malformed agentTurn records: "
                f"{invalid_session_targets} missing sessionTarget or invalid sessionTarget"
            )
        remaining_malformed = max(0, malformed_jobs - invalid_session_targets)
        if remaining_malformed > 0:
            issues.append(
                "[CRITICAL] Cron jobs contain additional malformed records: "
                f"{remaining_malformed}"
            )
    elif cron_summary["status"] == "WARN":
        warnings.append(
            "Cron jobs have structural warnings: "
            f"{cron_summary.get('malformed_job_count', 0)} malformed, "
            f"{cron_summary.get('delivery_fallback_ready_count', 0)} fallback-ready"
        )
    else:
        ok.append(
            "Cron jobs schema clean: "
            f"{cron_summary.get('jobs_total', 0)} jobs, "
            f"{cron_summary.get('delivery_fallback_ready_count', 0)} fallback-ready"
        )

    # agentToAgent
    a2a = tools_cfg.get("agentToAgent", {})
    a2a_enabled = a2a.get("enabled", False)
    if a2a_enabled:
        warnings.append("agentToAgent is enabled (increases cross-agent coupling)")
    else:
        ok.append("agentToAgent: disabled (isolated workloads)")

    # Loop detection (optional and version-dependent in OpenClaw builds)
    loop_det = tools_cfg.get("loopDetection")
    if isinstance(loop_det, dict):
        if loop_det.get("enabled", False):
            ok.append(
                "loopDetection: enabled "
                f"(warn={loop_det.get('warningThreshold')}, critical={loop_det.get('criticalThreshold')})"
            )
        else:
            warnings.append("loopDetection is configured but disabled")
    else:
        ok.append("loopDetection: not configured (optional / may be unsupported by current OpenClaw version)")

    # ===== REPORT =====
    print("=" * 65)
    print("  OpenClaw Configuration Validation Report")
    print("  Config: " + str(OPENCLAW_JSON))
    print("=" * 65)
    print()

    if issues:
        print(f"ISSUES ({len(issues)}):")
        for i in issues:
            print(f"  {i}")
        print()

    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  [!] {w}")
        print()

    print(f"PASSED ({len(ok)}):")
    for c in ok:
        print(f"  [OK] {c}")

    print()
    if issues:
        print(f"RESULT: FAIL ({len(issues)} issues, {len(warnings)} warnings)")
        return 1
    elif warnings:
        print(f"RESULT: PASS with {len(warnings)} warnings")
        return 0
    else:
        print("RESULT: PASS (all checks clean)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
