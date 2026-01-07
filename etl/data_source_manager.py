"""Data Source Manager for multi-source orchestration and failover support.

This module implements the DataSourceManager class that handles:
1. Dynamic data source selection based on configuration
2. Failover between multiple data sources
3. Parallel data fetching (future)
4. Unified interface for all data sources

Design Pattern: Strategy + Factory + Chain of Responsibility
- Strategy: Select appropriate data source based on configuration
- Factory: Instantiate extractors dynamically
- Chain: Failover through multiple sources on failure
Mathematical Foundation:
- Failover probability: P(success) = 1 - Pi(1 - p_i) for n sources
- Combined hit rate: eta_combined = Sigma(w_i x eta_i) for weighted sources
"""

import os
import yaml
import logging 
from typing import Dict, List, Optional, Any, Type, Iterable
from pathlib import Path
import pandas as pd
from importlib import import_module

from etl.base_extractor import BaseExtractor, ExtractorMetadata

logger = logging.getLogger(__name__)


class DataSourceManager:
    """Manager for multi-source data extraction with failover and selection strategies.

    This class orchestrates multiple data sources, handles failover, and provides
    a unified interface for data extraction regardless of the underlying source.

    Attributes:
        config: Configuration dictionary from data_sources_config.yml
        extractors: Dictionary of instantiated extractor instances
        active_extractor: Currently active data source extractor
    """

    def __init__(
        self,
        config_path: str = 'config/data_sources_config.yml',
        storage=None,
        execution_mode: str = "auto",
    ):
        """Initialize data source manager.

        Args:
            config_path: Path to data sources configuration file
            storage: DataStorage instance for cache operations

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration is invalid
        """
        self.config_path = Path(config_path)
        self.storage = storage
        self.execution_mode = str(execution_mode or "auto").lower()
        if self.execution_mode not in {"auto", "live", "synthetic"}:
            logger.warning("Unknown execution_mode=%s; defaulting to 'auto'", execution_mode)
            self.execution_mode = "auto"
        self.config = self._load_config()
        self.extractors: Dict[str, BaseExtractor] = {}
        self.active_extractor: Optional[BaseExtractor] = None

        # Initialize based on selection strategy
        self._initialize_extractors()

    def _load_config(self) -> Dict[str, Any]:
        """Load and validate data sources configuration.

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        if 'data_sources' not in config:
            raise ValueError("Invalid config: missing 'data_sources' section")

        logger.info(f"OK Loaded data sources configuration from {self.config_path}")
        return config['data_sources']

    def _get_enabled_providers(self) -> List[Dict[str, Any]]:
        """Get list of enabled data source providers sorted by priority.

        Returns:
            List of enabled provider configurations, sorted by priority (ascending)
        """
        providers = self.config.get('providers', [])
        enabled: List[Dict[str, Any]] = []
        enable_synthetic_env = os.getenv("ENABLE_SYNTHETIC_PROVIDER") or os.getenv("ENABLE_SYNTHETIC_DATA_SOURCE")
        synthetic_only = os.getenv("SYNTHETIC_ONLY")

        # Institutional guardrail: synthetic data must never be consumed implicitly during
        # live/auto runs. It is only admitted when explicitly requested (execution_mode=synthetic
        # or SYNTHETIC_ONLY) or when ENABLE_SYNTHETIC_PROVIDER is set for smoke/regression runs.
        synthetic_gate = bool(synthetic_only or self.execution_mode == "synthetic" or enable_synthetic_env)

        for provider in providers:
            name = str(provider.get('name') or "")
            if not name:
                continue

            if synthetic_only or self.execution_mode == "synthetic":
                if name == 'synthetic':
                    enabled.append({**provider, "enabled": True, "priority": 1})
                continue

            if name == "synthetic" and not synthetic_gate:
                continue

            is_enabled = provider.get('enabled', False)
            if name == "synthetic" and enable_synthetic_env:
                is_enabled = True
            if is_enabled:
                enabled.append(provider)

        # Sort by priority (lower number = higher priority)
        enabled.sort(key=lambda p: p.get('priority', 999))

        logger.info(
            "Found %s enabled providers: %s",
            len(enabled),
            ", ".join(str(p.get("name")) for p in enabled if p.get("name")),
        )
        return enabled

    def _resolve_provider_config_path(self, provider_config: Dict[str, Any]) -> Optional[str]:
        """Resolve config path for a provider, allowing env overrides.

        Env precedence:
        - <NAME>_CONFIG_PATH (generic)
        - SYNTHETIC_CONFIG_PATH (synthetic-specific convenience)
        Falls back to provider-configured config_file when present.
        """
        name = provider_config.get("name", "")
        env_override = os.getenv(f"{name.upper()}_CONFIG_PATH") if name else None
        if name == "synthetic":
            env_override = os.getenv("SYNTHETIC_CONFIG_PATH") or env_override
        return env_override or provider_config.get("config_file")

    def _instantiate_extractor(self, provider_config: Dict[str, Any]) -> Optional[BaseExtractor]:
        """Dynamically instantiate data source extractor.

        Args:
            provider_config: Provider configuration dictionary

        Returns:
            Instantiated extractor instance or None on failure
        """
        name = provider_config['name']
        config_path = self._resolve_provider_config_path(provider_config)
        base_kwargs: Dict[str, Any] = {"name": name, "storage": self.storage}
        if config_path:
            base_kwargs["config_path"] = config_path

        try:
            # Get adapter class path from config
            adapter_registry = self.config.get('adapters', {}).get('adapter_registry', {})
            adapter_path = adapter_registry.get(name)

            if not adapter_path:
                logger.error(f"No adapter registered for '{name}'")
                return None

            # Dynamically import extractor class
            # Format: "etl.yfinance_extractor.YFinanceExtractor"
            module_path, class_name = adapter_path.rsplit('.', 1)
            module = import_module(module_path)
            extractor_class: Type[BaseExtractor] = getattr(module, class_name)

            # Check if API key is required
            credentials_env = provider_config.get('credentials_env')
            if credentials_env:
                # SECURITY: Use secret_loader for secure secret management
                # Supports both Docker secrets and environment variables
                from etl.secret_loader import load_secret
                api_key = load_secret(credentials_env)
                if not api_key:
                    logger.info(
                        "Skipping %s extractor; credential %s not configured. "
                        "Set the environment variable or Docker secret to enable this provider.",
                        name,
                        credentials_env,
                    )
                    return None

                # Instantiate with API key
                extractor = extractor_class(api_key=api_key, **base_kwargs)
            else:
                # Instantiate without API key (e.g., yfinance)
                extractor = extractor_class(**base_kwargs)

            logger.info(f"OK Instantiated {name} extractor")
            return extractor

        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import extractor for '{name}': {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to instantiate extractor for '{name}': {e}")
            return None

    def _initialize_extractors(self) -> None:
        """Initialize extractors based on selection strategy.

        Selection modes:
        - priority: Use highest priority enabled source
        - fallback: Try sources in priority order until success
        - parallel: Use multiple sources concurrently (future)
        """
        selection_config = self.config.get('selection_strategy', {})
        mode = selection_config.get('mode', 'priority')

        enabled_providers = self._get_enabled_providers()

        if not enabled_providers:
            logger.error("No enabled data sources found in configuration")
            raise ValueError("No enabled data sources available")

        if mode == 'priority':
            # Load only the highest priority source
            primary = enabled_providers[0]
            extractor = self._instantiate_extractor(primary)

            if extractor:
                self.extractors[primary['name']] = extractor
                self.active_extractor = extractor
                logger.info(f"OK Using primary data source: {primary['name']}")
            else:
                raise RuntimeError(f"Failed to initialize primary source: {primary['name']}")

        elif mode == 'fallback':
            # Load all enabled sources for failover
            for provider in enabled_providers:
                extractor = self._instantiate_extractor(provider)
                if extractor:
                    self.extractors[provider['name']] = extractor

            if not self.extractors:
                raise RuntimeError("Failed to initialize any data sources")

            # Set primary as active
            self.active_extractor = list(self.extractors.values())[0]
            logger.info(f"OK Initialized {len(self.extractors)} sources for failover")
            logger.info(f"  Primary: {self.active_extractor.name}")

        elif mode == 'parallel':
            # Future: Load all sources for parallel fetching
            logger.warning("Parallel mode not yet implemented, falling back to priority mode")
            primary = enabled_providers[0]
            extractor = self._instantiate_extractor(primary)
            if extractor:
                self.extractors[primary['name']] = extractor
                self.active_extractor = extractor
                logger.info(f"OK Using primary data source: {primary['name']}")
            else:
                raise RuntimeError(f"Failed to initialize primary source: {primary['name']}")

        else:
            raise ValueError(f"Unknown selection mode: {mode}")

    def extract_ohlcv(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        prefer_source: Optional[str] = None,
        chunk_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """Extract OHLCV data using active source with optional failover.

        Args:
            tickers: List of ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            prefer_source: Optional preferred source name
            chunk_size: Optional max tickers per batch to limit memory footprint;
                when set and tickers exceed this size, extraction runs in batches
                and concatenates results. Defaults to None (no chunking).

        Returns:
            DataFrame with OHLCV data

        Raises:
            RuntimeError: If all sources fail
        """
        selection_config = self.config.get('selection_strategy', {})
        mode = selection_config.get('mode', 'priority')

        # Determine which extractor to use
        if prefer_source and prefer_source in self.extractors:
            extractor = self.extractors[prefer_source]
            logger.info(f"Using preferred source: {prefer_source}")
        else:
            extractor = self.active_extractor

        effective_chunk = chunk_size
        env_chunk = os.getenv("DATA_SOURCE_CHUNK_SIZE")
        if effective_chunk is None and env_chunk and env_chunk.isdigit():
            effective_chunk = int(env_chunk)
        if effective_chunk is not None and effective_chunk <= 0:
            effective_chunk = None

        def _memory_mb(df: pd.DataFrame) -> float:
            try:
                return float(df.memory_usage(deep=True).sum()) / 1_000_000.0
            except Exception:
                return 0.0

        def _extract_batch(batch: Iterable[str]) -> pd.DataFrame:
            try:
                data = extractor.extract_ohlcv(list(batch), start_date, end_date)
                if data is None or data.empty:
                    raise RuntimeError(f"{extractor.name} returned empty data")
                logger.info("OK Extracted %s rows from %s", len(data), extractor.name)
                mem_mb = _memory_mb(data)
                if mem_mb > 0:
                    logger.debug("Batch memory usage approx %.1f MB (%s)", mem_mb, extractor.name)
                return data
            except Exception as e:
                logger.error(f"FAIL Extraction failed from {extractor.name}: {e}")
                if mode == 'fallback':
                    return self._failover_extraction(list(batch), start_date, end_date,
                                                    failed_source=extractor.name)
                raise

        if effective_chunk and len(tickers) > effective_chunk:
            all_frames: List[pd.DataFrame] = []
            for idx, i in enumerate(range(0, len(tickers), effective_chunk)):
                batch = tickers[i : i + effective_chunk]
                logger.info("Chunked OHLCV extraction batch %s: %s", idx + 1, batch)
                batch_df = _extract_batch(batch)
                if batch_df is not None and not batch_df.empty:
                    all_frames.append(batch_df)
            if not all_frames:
                return pd.DataFrame()
            combined = pd.concat(all_frames, ignore_index=False)
            return combined

        return _extract_batch(tickers)

    def _failover_extraction(self, tickers: List[str], start_date: str,
                            end_date: str, failed_source: str) -> pd.DataFrame:
        """Attempt extraction from fallback sources.

        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            failed_source: Name of source that failed

        Returns:
            DataFrame with OHLCV data

        Raises:
            RuntimeError: If all sources fail
        """
        failover_config = self.config.get('failover', {})
        max_attempts = failover_config.get('max_failover_attempts', 3)

        logger.info(f"Attempting failover (max {max_attempts} attempts)...")

        # Try remaining sources in priority order
        for source_name, extractor in self.extractors.items():
            if source_name == failed_source:
                continue  # Skip the failed source

            if max_attempts <= 0:
                break

            try:
                logger.info(f"Trying fallback source: {source_name}")
                data = extractor.extract_ohlcv(tickers, start_date, end_date)

                if data is not None and not data.empty:
                    logger.info(f"OK Failover successful: {source_name} returned {len(data)} rows")
                    self.active_extractor = extractor  # Switch active extractor
                    return data

            except Exception as e:
                logger.warning(f"Failover attempt failed for {source_name}: {e}")
                max_attempts -= 1

        # All sources failed
        raise RuntimeError(f"All data sources failed after {max_attempts} attempts")

    def get_extractor(self, source_name: str) -> Optional[BaseExtractor]:
        """Get specific extractor by name.

        Args:
            source_name: Name of data source

        Returns:
            Extractor instance or None if not found
        """
        return self.extractors.get(source_name)

    def get_active_source(self) -> str:
        """Get name of currently active data source.

        Returns:
            Name of active source
        """
        return self.active_extractor.name if self.active_extractor else "None"

    def get_available_sources(self) -> List[str]:
        """Get list of available (instantiated) data sources.

        Returns:
            List of source names
        """
        return list(self.extractors.keys())

    def get_cache_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get cache statistics for all extractors.

        Returns:
            Dictionary mapping source names to their cache statistics
        """
        stats = {}
        for name, extractor in self.extractors.items():
            stats[name] = extractor.get_cache_statistics()
        return stats

    def validate_data(self, data: pd.DataFrame, source_name: Optional[str] = None) -> Dict[str, Any]:
        """Validate data using appropriate extractor's validation method.

        Args:
            data: DataFrame to validate
            source_name: Optional specific source to use for validation

        Returns:
            Validation report dictionary
        """
        if source_name and source_name in self.extractors:
            extractor = self.extractors[source_name]
        else:
            extractor = self.active_extractor

        if not extractor:
            raise RuntimeError("No active extractor available for validation")

        return extractor.validate_data(data)

    def __repr__(self) -> str:
        """String representation of manager."""
        return (f"DataSourceManager(sources={len(self.extractors)}, "
                f"active='{self.get_active_source()}')")

    def __str__(self) -> str:
        """Human-readable string representation."""
        active = self.get_active_source()
        sources = ', '.join(self.extractors.keys())
        return (f"Data Source Manager\n"
                f"  Available sources: {sources}\n"
                f"  Active source: {active}\n"
                f"  Mode: {self.config.get('selection_strategy', {}).get('mode', 'unknown')}")
