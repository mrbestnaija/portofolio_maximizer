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
- Failover probability: P(success) = 1 - ∏(1 - p_i) for n sources
- Combined hit rate: η_combined = Σ(w_i × η_i) for weighted sources
"""

import os
import yaml
import logging
from typing import Dict, List, Optional, Any, Type
from pathlib import Path
import pandas as pd
from importlib import import_module

from etl.base_extractor import BaseExtractor, ExtractorMetadata

logging.basicConfig(level=logging.INFO)
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

    def __init__(self, config_path: str = 'config/data_sources_config.yml',
                 storage=None):
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

        logger.info(f"✓ Loaded data sources configuration from {self.config_path}")
        return config['data_sources']

    def _get_enabled_providers(self) -> List[Dict[str, Any]]:
        """Get list of enabled data source providers sorted by priority.

        Returns:
            List of enabled provider configurations, sorted by priority (ascending)
        """
        providers = self.config.get('providers', [])
        enabled = [p for p in providers if p.get('enabled', False)]

        # Sort by priority (lower number = higher priority)
        enabled.sort(key=lambda p: p.get('priority', 999))

        logger.info(f"Found {len(enabled)} enabled providers: "
                   f"{', '.join(p['name'] for p in enabled)}")
        return enabled

    def _instantiate_extractor(self, provider_config: Dict[str, Any]) -> Optional[BaseExtractor]:
        """Dynamically instantiate data source extractor.

        Args:
            provider_config: Provider configuration dictionary

        Returns:
            Instantiated extractor instance or None on failure
        """
        name = provider_config['name']

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
                api_key = os.getenv(credentials_env)
                if not api_key:
                    logger.warning(f"Missing API key for {name}: {credentials_env} not set in .env")
                    return None

                # Instantiate with API key
                extractor = extractor_class(
                    name=name,
                    api_key=api_key,
                    storage=self.storage
                )
            else:
                # Instantiate without API key (e.g., yfinance)
                extractor = extractor_class(
                    name=name,
                    storage=self.storage
                )

            logger.info(f"✓ Instantiated {name} extractor")
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
                logger.info(f"✓ Using primary data source: {primary['name']}")
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
            logger.info(f"✓ Initialized {len(self.extractors)} sources for failover")
            logger.info(f"  Primary: {self.active_extractor.name}")

        elif mode == 'parallel':
            # Future: Load all sources for parallel fetching
            logger.warning("Parallel mode not yet implemented, falling back to priority mode")
            self._initialize_extractors()  # Recursively call with default mode

        else:
            raise ValueError(f"Unknown selection mode: {mode}")

    def extract_ohlcv(self, tickers: List[str], start_date: str,
                      end_date: str, prefer_source: Optional[str] = None) -> pd.DataFrame:
        """Extract OHLCV data using active source with optional failover.

        Args:
            tickers: List of ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            prefer_source: Optional preferred source name

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

        # Attempt extraction
        try:
            data = extractor.extract_ohlcv(tickers, start_date, end_date)

            if data is None or data.empty:
                raise RuntimeError(f"{extractor.name} returned empty data")

            logger.info(f"✓ Successfully extracted {len(data)} rows from {extractor.name}")
            return data

        except Exception as e:
            logger.error(f"✗ Extraction failed from {extractor.name}: {e}")

            # Attempt failover if enabled
            if mode == 'fallback':
                return self._failover_extraction(tickers, start_date, end_date,
                                                failed_source=extractor.name)
            else:
                raise RuntimeError(f"Data extraction failed: {e}")

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
                    logger.info(f"✓ Failover successful: {source_name} returned {len(data)} rows")
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
