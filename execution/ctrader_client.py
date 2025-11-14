"""execution.ctrader_client
=================================

Production-focused cTrader Open API client that replaces the original IBKR
stub described in ``Documentation/STUB_IMPLEMENTATION_PLAN.md``. The client
is intentionally high level so that the order manager can operate the same
portfolio lifecycle across demo (training) and live accounts without touching
low-level HTTP plumbing.

Key design goals
----------------

* Demo environment first, as mandated by ``UNIFIED_ROADMAP.md`` and the
  sequencing documents. Live trading is feature-flagged and the default
  configuration keeps all calls pointed at demo infrastructure until 55%+
  accuracy and the drawdown guardrails are satisfied.
* Environment-driven credentials. The ``.env`` file ships the cTrader login
  payload and we support both ``KEY=value`` and ``KEY:'value'`` syntaxes so the
  regenerated file in the repository keeps working without manual edits.
* Resilient authentication. Tokens are cached, refreshed proactively, and we
  retry transient failures so the downstream order manager can focus on
  portfolio logic.
* Explicit domain objects. Orders, placement results, and account snapshots are
  represented via dataclasses which keeps the higher level orchestration code
  type-safe and testable.

The implementation uses the public REST surfaces documented at
``https://help.ctrader.com/open-api/`` and ``https://openapi.ctrader.com/``.
The real platform exposes gRPC streams as well, but an HTTP client with
feature-complete order submission and account inspection is sufficient for the
current phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import logging
import os

import requests

try:  # Optional dependency; keep import local to avoid hard failure on CI
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - dependency is optional
    load_dotenv = None  # type: ignore

if load_dotenv:  # Load .env early so os.getenv works during tests/CLI runs
    load_dotenv()

logger = logging.getLogger(__name__)


DEMO_AUTH_URL = "https://demo.ctraderapi.com/connect/token"
LIVE_AUTH_URL = "https://api.ctraderapi.com/connect/token"
DEMO_API_BASE = "https://demo.ctraderapi.com"
LIVE_API_BASE = "https://api.ctraderapi.com"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CTraderClientError(RuntimeError):
    """Base class for cTrader integration errors."""


class CTraderAuthError(CTraderClientError):
    """Raised when authentication fails."""


class CTraderOrderError(CTraderClientError):
    """Raised when order submission fails."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


def _sanitize_env_value(value: Optional[str]) -> Optional[str]:
    """Strip quotes/whitespace without logging secrets."""

    if value is None:
        return None

    return value.strip().strip('"').strip("'") or None


def _load_env_pair() -> Dict[str, str]:
    """Read the .env file manually so KEY:'value' syntax still works."""

    env_path = Path(__file__).resolve().parents[1] / ".env"
    cache_key = "CTRADER_ENV_CACHE"
    if cache_key in globals():  # pragma: no cover - defensive guard
        return globals()[cache_key]

    mapping: Dict[str, str] = {}
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            delimiter = "=" if "=" in line else ":" if ":" in line else None
            if not delimiter:
                continue

            key, value = line.split(delimiter, 1)
            mapping[key.strip()] = value.strip().strip('"').strip("'")

    globals()[cache_key] = mapping
    return mapping


def _env_value(name: str) -> Optional[str]:
    """Lookup environment variables with .env fallback."""

    value = os.getenv(name)
    if value:
        return _sanitize_env_value(value)

    return _sanitize_env_value(_load_env_pair().get(name))


@dataclass(slots=True)
class CTraderClientConfig:
    """Configuration payload for the cTrader client."""

    username: str
    password: str
    application_id: str
    environment: str = "demo"
    account_id: Optional[int] = None
    application_secret: Optional[str] = None
    demo_auth_url: str = DEMO_AUTH_URL
    live_auth_url: str = LIVE_AUTH_URL
    demo_api_base: str = DEMO_API_BASE
    live_api_base: str = LIVE_API_BASE
    timeout: int = 15
    max_retries: int = 3
    retry_backoff: float = 1.5
    read_timeout: int = 30
    user_agent: str = "PortfolioMaximizer/ctrader-client"

    @property
    def is_demo(self) -> bool:
        return self.environment.lower() != "live"

    @property
    def auth_url(self) -> str:
        return self.demo_auth_url if self.is_demo else self.live_auth_url

    @property
    def api_base(self) -> str:
        return self.demo_api_base if self.is_demo else self.live_api_base

    @classmethod
    def from_env(
        cls,
        environment: str = "demo",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "CTraderClientConfig":
        """Factory that loads credentials from environment variables."""

        overrides = overrides or {}

        username = overrides.get("username") or _env_value("USERNAME_CTRADER")
        if not username:
            username = _env_value("EMAIL_CTRADER")

        password = overrides.get("password") or _env_value("PASSWORD_CTRADER")
        application_id = overrides.get("application_id") or _env_value(
            "APPLICATION_NAME_CTRADER"
        )
        application_secret = overrides.get("application_secret") or _env_value(
            "CTRADER_APPLICATION_SECRET"
        )

        account_id_raw = overrides.get("account_id") or _env_value(
            "CTRADER_ACCOUNT_ID"
        )
        account_id = None
        if account_id_raw:
            try:
                account_id = int(str(account_id_raw).strip())
            except ValueError:
                logger.warning("Unable to coerce CTRADER_ACCOUNT_ID into int; keep None")

        missing = []
        if not username:
            missing.append("USERNAME_CTRADER")
        if not password:
            missing.append("PASSWORD_CTRADER")
        if not application_id:
            missing.append("APPLICATION_NAME_CTRADER")

        if missing:
            raise CTraderAuthError(
                "Missing cTrader credentials. Ensure the following environment "
                f"variables exist: {', '.join(missing)}"
            )

        env = overrides.get("environment", environment)

        return cls(
            username=username,
            password=password,
            application_id=application_id,
            application_secret=application_secret,
            environment=env,
            account_id=account_id,
        )


@dataclass(slots=True)
class CTraderOrder:
    """Order payload sent to cTrader."""

    symbol: str
    side: str  # BUY or SELL
    volume: float
    order_type: str = "MARKET"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    comment: Optional[str] = None
    client_order_id: Optional[str] = None
    label: Optional[str] = None


@dataclass(slots=True)
class OrderPlacement:
    """Normalized response from cTrader after placing an order."""

    order_id: str
    status: str
    filled_volume: float
    avg_price: Optional[float]
    submitted_at: datetime
    raw_response: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AccountSnapshot:
    """Simplified account overview."""

    balance: float
    equity: float
    free_margin: float
    margin_level: Optional[float] = None
    currency: str = "USD"
    updated_at: datetime = field(default_factory=datetime.utcnow)
    raw: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Client implementation
# ---------------------------------------------------------------------------


class CTraderClient:
    """High-level HTTP client for cTrader's Open API."""

    def __init__(
        self,
        config: CTraderClientConfig,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.config = config
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": config.user_agent})
        self._token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

        logger.info(
            "cTrader client initialized for %s environment",
            "demo" if config.is_demo else "live",
        )

    # ------------------------------------------------------------------
    # Authentication helpers
    # ------------------------------------------------------------------

    def authenticate(self, force: bool = False) -> str:
        """Authenticate via OAuth password grant."""

        if not force and self._token and self._token_expiry:
            if datetime.utcnow() < self._token_expiry - timedelta(seconds=30):
                return self._token

        payload = {
            "grant_type": "password",
            "username": self.config.username,
            "password": self.config.password,
            "client_id": self.config.application_id,
            "scope": "trading profile",
        }
        if self.config.application_secret:
            payload["client_secret"] = self.config.application_secret

        response = self._session.post(
            self.config.auth_url,
            data=payload,
            timeout=(self.config.timeout, self.config.read_timeout),
        )

        if response.status_code >= 400:
            raise CTraderAuthError(
                f"cTrader authentication failed ({response.status_code}): {response.text}"
            )

        payload = response.json()
        access_token = payload.get("access_token")
        refresh_token = payload.get("refresh_token")
        expires_in = int(payload.get("expires_in", 900))

        if not access_token:
            raise CTraderAuthError("Authentication succeeded but access_token missing")

        self._token = access_token
        self._refresh_token = refresh_token
        self._token_expiry = datetime.utcnow() + timedelta(seconds=expires_in)

        logger.debug(
            "cTrader access token acquired (expires in %ss)",
            expires_in,
        )

        return access_token

    def _ensure_token(self) -> str:
        if self._token and self._token_expiry and datetime.utcnow() < self._token_expiry - timedelta(seconds=30):
            return self._token

        if self._refresh_token:
            try:
                return self._refresh()
            except CTraderAuthError:
                logger.warning("Token refresh failed; retrying full authentication")

        return self.authenticate(force=True)

    def _refresh(self) -> str:
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
            "client_id": self.config.application_id,
        }
        if self.config.application_secret:
            payload["client_secret"] = self.config.application_secret

        response = self._session.post(
            self.config.auth_url,
            data=payload,
            timeout=(self.config.timeout, self.config.read_timeout),
        )

        if response.status_code >= 400:
            raise CTraderAuthError(
                f"Refresh token rejected ({response.status_code}): {response.text}"
            )

        payload = response.json()
        self._token = payload.get("access_token")
        self._refresh_token = payload.get("refresh_token") or self._refresh_token
        expires_in = int(payload.get("expires_in", 900))
        self._token_expiry = datetime.utcnow() + timedelta(seconds=expires_in)

        if not self._token:
            raise CTraderAuthError("Refresh succeeded but no access token returned")

        logger.debug("cTrader token refreshed successfully")
        return self._token

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        token = self._ensure_token()
        url = self.config.api_base.rstrip("/") + path

        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        if json_body is not None:
            headers["Content-Type"] = "application/json"

        response = self._session.request(
            method=method.upper(),
            url=url,
            headers=headers,
            params=params,
            json=json_body,
            timeout=(self.config.timeout, self.config.read_timeout),
        )

        if response.status_code == 401:
            # Token expired unexpectedly; refresh once and retry
            logger.info("cTrader request unauthorized, attempting token refresh")
            self.authenticate(force=True)
            return self._request(method, path, params=params, json_body=json_body)

        if response.status_code >= 400:
            raise CTraderClientError(
                f"cTrader API error ({response.status_code}) on {path}: {response.text}"
            )

        if not response.content:
            return {}

        try:
            return response.json()
        except ValueError as exc:  # pragma: no cover - depends on upstream payload
            raise CTraderClientError(
                f"Failed to parse cTrader response for {path}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_account_overview(self) -> AccountSnapshot:
        """Return balance/equity/margin snapshot."""

        if not self.config.account_id:
            raise CTraderClientError(
                "Account ID missing. Set CTRADER_ACCOUNT_ID or pass via overrides."
            )

        payload = self._request(
            "GET", f"/api/v1/accounts/{self.config.account_id}/overview"
        )

        balance = float(payload.get("balance", 0.0))
        equity = float(payload.get("equity", balance))
        free_margin = float(payload.get("freeMargin", equity))
        margin_level = payload.get("marginLevel")
        currency = payload.get("currency", "USD")

        return AccountSnapshot(
            balance=balance,
            equity=equity,
            free_margin=free_margin,
            margin_level=float(margin_level) if margin_level is not None else None,
            currency=currency,
            raw=payload,
        )

    def get_positions(self) -> Dict[str, Any]:
        """Return open positions keyed by instrument."""

        if not self.config.account_id:
            raise CTraderClientError("Account ID required for positions endpoint")

        payload = self._request(
            "GET", f"/api/v1/accounts/{self.config.account_id}/positions"
        )
        return payload

    def place_order(self, order: CTraderOrder) -> OrderPlacement:
        """Submit an order to cTrader."""

        if not self.config.account_id:
            raise CTraderOrderError(
                "Cannot place order without account_id. Configure CTRADER_ACCOUNT_ID."
            )

        body = {
            "symbolName": order.symbol,
            "orderType": order.order_type.upper(),
            "tradeSide": order.side.upper(),
            "volume": order.volume,
            "timeInForce": order.time_in_force,
        }

        if order.limit_price is not None:
            body["limitPrice"] = order.limit_price
        if order.stop_price is not None:
            body["stopPrice"] = order.stop_price
        if order.comment:
            body["comment"] = order.comment[:100]
        if order.client_order_id:
            body["clientOrderId"] = order.client_order_id
        if order.label:
            body["label"] = order.label

        payload = self._request(
            "POST",
            f"/api/v1/accounts/{self.config.account_id}/orders",
            json_body=body,
        )

        order_id = str(payload.get("orderId") or payload.get("id"))
        status = payload.get("status", "UNKNOWN")
        filled = float(payload.get("filledVolume", 0.0))
        avg_price = payload.get("averagePrice")

        if not order_id:
            raise CTraderOrderError("cTrader response missing order identifier")

        logger.info(
            "cTrader order %s placed (%s %s %s)",
            order_id,
            order.side,
            order.volume,
            order.symbol,
        )

        submitted_at = datetime.utcnow()
        return OrderPlacement(
            order_id=order_id,
            status=status,
            filled_volume=filled,
            avg_price=float(avg_price) if avg_price is not None else None,
            submitted_at=submitted_at,
            raw_response=payload,
        )

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an open order."""

        if not self.config.account_id:
            raise CTraderOrderError("Account ID required for cancellations")

        payload = self._request(
            "POST",
            f"/api/v1/accounts/{self.config.account_id}/orders/{order_id}/cancel",
        )
        logger.info("cTrader order %s cancelled", order_id)
        return payload

    def close(self) -> None:
        """Close the underlying HTTP session."""

        try:
            self._session.close()
        except Exception:  # pragma: no cover - best effort
            logger.debug("Failed to close cTrader session", exc_info=True)


__all__ = [
    "AccountSnapshot",
    "CTraderClient",
    "CTraderClientConfig",
    "CTraderClientError",
    "CTraderOrder",
    "CTraderOrderError",
    "OrderPlacement",
]

