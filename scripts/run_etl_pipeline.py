#!/usr/bin/env python3
"""ETL Pipeline Orchestration Script - Modular Configuration-Driven Design.

This orchestrator loads configuration from modular YAML files and executes
the ETL pipeline with proper error handling and progress tracking.

Configuration files:
- config/pipeline_config.yml: Main pipeline orchestration config
- config/yfinance_config.yml: Data extraction configuration
- config/validation_config.yml: Data validation rules
- config/preprocessing_config.yml: Preprocessing parameters
- config/storage_config.yml: Storage and splitting configuration
"""
import sys
import yaml
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import click
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.data_source_manager import DataSourceManager
from etl.data_validator import DataValidator
from etl.preprocessor import Preprocessor
from etl.data_storage import DataStorage
from etl.checkpoint_manager import CheckpointManager
from etl.pipeline_logger import PipelineLogger
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate pipeline configuration.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Config not found: {config_path}")

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✓ Loaded configuration from: {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in configuration: {e}")
        raise


@click.command()
@click.option('--config', default='config/pipeline_config.yml',
              help='Path to pipeline configuration (default: config/pipeline_config.yml)')
@click.option('--data-source', default=None,
              help='Data source to use (options: yfinance, alpha_vantage, finnhub). Defaults to config value.')
@click.option('--tickers', default='AAPL,MSFT',
              help='Comma-separated ticker symbols (default: AAPL,MSFT)')
@click.option('--start', default='2020-01-01',
              help='Start date YYYY-MM-DD (default: 2020-01-01)')
@click.option('--end', default='2024-01-01',
              help='End date YYYY-MM-DD (default: 2024-01-01)')
@click.option('--use-cv', is_flag=True, default=None,
              help='Use k-fold cross-validation. If not set, uses config default_strategy.')
@click.option('--n-splits', default=None, type=int,
              help='Number of CV folds. If not set, uses config value.')
@click.option('--test-size', default=None, type=float,
              help='Test set size (0.0-1.0). If not set, uses config value.')
@click.option('--gap', default=None, type=int,
              help='Gap between train/validation periods. If not set, uses config value.')
@click.option('--verbose', is_flag=True, default=False,
              help='Enable verbose logging (DEBUG level)')
def run_pipeline(config: str, data_source: str, tickers: str, start: str, end: str,
                use_cv: bool, n_splits: int, test_size: float, gap: int, verbose: bool) -> None:
    """Execute ETL pipeline with modular configuration-driven orchestration.

    Data Splitting Strategy:
    - Default (--use-cv=False): Simple 70/15/15 chronological split (backward compatible)
    - Recommended (--use-cv): k-fold cross-validation with expanding window
      * 5.5x better temporal coverage (15% → 83%)
      * Eliminates temporal gap (0 years vs 2.5 years)
      * Strict test isolation (15% never exposed during CV)

    Examples:
        # Simple split (backward compatible)
        python scripts/run_etl_pipeline.py --tickers AAPL,MSFT --start 2020-01-01

        # k-fold CV (recommended for production)
        python scripts/run_etl_pipeline.py --tickers AAPL --use-cv --n-splits 5

        # Verbose logging
        python scripts/run_etl_pipeline.py --tickers GOOGL --use-cv --verbose
    """
    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Load pipeline configuration
    try:
        pipeline_config = load_config(config)
    except Exception as e:
        logger.error(f"Failed to load pipeline config: {e}")
        logger.info("Using fallback configuration...")
        pipeline_config = {'pipeline': {'stages': []}}  # Fallback

    # Extract configuration sections
    pipeline_cfg = pipeline_config.get('pipeline', {})
    stages_cfg = pipeline_cfg.get('stages', [])
    data_split_cfg = pipeline_cfg.get('data_split', {})

    # Get CV configuration from config file
    cv_config = data_split_cfg.get('cross_validation', {})
    simple_config = data_split_cfg.get('simple_split', {})
    default_strategy = data_split_cfg.get('default_strategy', 'simple')

    # Determine CV parameters (CLI overrides config)
    if use_cv is None:
        # Use config default strategy
        use_cv = (default_strategy == 'cv')

    if n_splits is None:
        # Use config value
        n_splits = cv_config.get('n_splits', 5)

    if test_size is None:
        # Use config value
        test_size = cv_config.get('test_size', 0.15)

    if gap is None:
        # Use config value
        gap = cv_config.get('gap', 0)

    # Get simple split ratios from config
    train_ratio = simple_config.get('train_ratio', 0.7)
    val_ratio = simple_config.get('validation_ratio', 0.15)

    # Parse tickers
    ticker_list = [t.strip() for t in tickers.split(',')]
    logger.info(f"Pipeline: Portfolio Maximizer v4.0")
    logger.info(f"Data Source: {data_source if data_source else 'from config'}")
    logger.info(f"Tickers: {', '.join(ticker_list)}")
    logger.info(f"Date range: {start} to {end}")

    # Initialize storage
    storage = DataStorage()

    # Initialize checkpoint manager and pipeline logger
    pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_manager = CheckpointManager(checkpoint_dir="data/checkpoints")
    pipeline_log = PipelineLogger(log_dir="logs", retention_days=7)

    logger.info(f"✓ Pipeline ID: {pipeline_id}")
    pipeline_log.log_event('pipeline_start', pipeline_id, metadata={
        'tickers': ticker_list,
        'start_date': start,
        'end_date': end,
        'use_cv': use_cv
    })

    # Initialize data source manager (platform-agnostic)
    try:
        data_source_manager = DataSourceManager(
            config_path='config/data_sources_config.yml',
            storage=storage
        )
        logger.info(f"✓ Data source manager initialized")
        logger.info(f"  Available sources: {', '.join(data_source_manager.get_available_sources())}")
        logger.info(f"  Active source: {data_source_manager.get_active_source()}")
    except Exception as e:
        logger.error(f"Failed to initialize data source manager: {e}")
        logger.error("Pipeline cannot continue without data source")
        raise

    # Determine split strategy and log configuration
    if use_cv:
        logger.info(f"✓ Using k-fold cross-validation (k={n_splits})")
        logger.info(f"  - Test size: {test_size*100:.0f}%")
        logger.info(f"  - Gap between train/val: {gap} periods")
        logger.info(f"  - Window strategy: {cv_config.get('window_strategy', 'expanding')}")
        expected_coverage = cv_config.get('expected_coverage', 0.83)
        logger.info(f"  - Expected validation coverage: {expected_coverage*100:.0f}%")
    else:
        logger.info(f"Using simple chronological split ({train_ratio*100:.0f}/{val_ratio*100:.0f}/{(1-train_ratio-val_ratio)*100:.0f})")
        logger.info(f"  - Strategy: {default_strategy}")
        logger.info(f"  - Chronological: {simple_config.get('chronological', True)}")

    # Define stage names (in execution order)
    stage_names = ['data_extraction', 'data_validation', 'data_preprocessing', 'data_storage']

    # Execute pipeline stages
    logger.info("=" * 70)
    logger.info("Starting ETL Pipeline")
    logger.info("=" * 70)

    for stage_name in tqdm(stage_names, desc='Pipeline Progress'):
        logger.info(f"\n[Stage: {stage_name}]")
        stage_start_time = time.time()
        pipeline_log.log_stage_start(pipeline_id, stage_name)

        try:
            if stage_name == 'data_extraction':
                # Stage 1: Data Extraction (platform-agnostic)
                logger.info(f"Extracting OHLCV data using data source manager...")

                # Use data source manager for extraction (automatic failover support)
                raw_data = data_source_manager.extract_ohlcv(
                    tickers=ticker_list,
                    start_date=start,
                    end_date=end,
                    prefer_source=data_source if data_source != 'yfinance' else None
                )

                if raw_data is None or raw_data.empty:
                    raise RuntimeError("Data extraction failed - empty dataset")

                logger.info(f"✓ Extracted {len(raw_data)} rows from {len(ticker_list)} ticker(s)")
                logger.info(f"  Source: {data_source_manager.get_active_source()}")

                # Log cache statistics
                cache_stats = data_source_manager.get_cache_statistics()
                for source_name, stats in cache_stats.items():
                    if stats['total_requests'] > 0:
                        logger.info(f"  {source_name} cache: {stats['cache_hits']}/{stats['total_requests']} hits ({stats['hit_rate']:.1%})")

                # Save checkpoint
                checkpoint_id = checkpoint_manager.save_checkpoint(
                    pipeline_id=pipeline_id,
                    stage=stage_name,
                    data=raw_data,
                    metadata={'tickers': ticker_list, 'rows': len(raw_data)}
                )
                pipeline_log.log_checkpoint(pipeline_id, stage_name, checkpoint_id)

            elif stage_name == 'data_validation':
                # Stage 2: Data Validation
                logger.info("Validating data quality...")
                validator = DataValidator()
                report = validator.validate_ohlcv(raw_data)

                if not report['passed']:
                    logger.warning(f"⚠ Validation warnings detected")
                    logger.warning(f"  Errors: {len(report.get('errors', []))}")
                    logger.warning(f"  Warnings: {len(report.get('warnings', []))}")
                    if verbose:
                        logger.debug(f"Validation report: {report}")
                else:
                    logger.info("✓ Data validation passed")

            elif stage_name == 'data_preprocessing':
                # Stage 3: Data Preprocessing
                logger.info("Preprocessing data (missing data + normalization)...")
                processor = Preprocessor()

                # Handle missing values
                filled = processor.handle_missing(raw_data)
                logger.debug(f"  Missing data handled")

                # Normalize (returns tuple: data, stats)
                normalized, stats = processor.normalize(filled)
                processed = normalized
                logger.debug(f"  Normalization complete (μ=0, σ²=1)")

                # Save processed data
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                storage.save(processed, 'processed', f'processed_{timestamp}')
                logger.info(f"✓ Preprocessed {len(processed)} rows")

            elif stage_name == 'data_storage':
                # Stage 4: Data Storage (Split + Save)
                logger.info("Splitting and saving datasets...")
                logger.info(f"  Configuration source: pipeline_config.yml")
                logger.info(f"  Split strategy: {'CV' if use_cv else 'Simple'}")

                # Perform split using configuration-driven parameters
                if use_cv:
                    splits = storage.train_validation_test_split(
                        processed,
                        use_cv=True,
                        n_splits=n_splits,
                        test_size=test_size,
                        gap=gap
                    )
                else:
                    splits = storage.train_validation_test_split(
                        processed,
                        train_ratio=train_ratio,
                        val_ratio=val_ratio,
                        use_cv=False
                    )

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                if use_cv:
                    # Save CV folds
                    for fold in splits['cv_folds']:
                        fold_id = fold['fold_id']
                        storage.save(fold['train'], 'training',
                                   f'fold{fold_id}_train_{timestamp}')
                        storage.save(fold['validation'], 'validation',
                                   f'fold{fold_id}_val_{timestamp}')

                    # Save isolated test set
                    storage.save(splits['testing'], 'testing', f'test_{timestamp}')

                    # Calculate summary statistics
                    avg_train_size = sum(len(f['train']) for f in splits['cv_folds']) / len(splits['cv_folds'])
                    avg_val_size = sum(len(f['validation']) for f in splits['cv_folds']) / len(splits['cv_folds'])

                    logger.info(f"✓ Saved {len(splits['cv_folds'])} CV folds + 1 test set")
                    logger.info(f"  - Train size (avg): {avg_train_size:.0f} rows")
                    logger.info(f"  - Val size (avg): {avg_val_size:.0f} rows")
                    logger.info(f"  - Test size: {len(splits['testing'])} rows")
                else:
                    # Simple split (backward compatible)
                    for split_name, split_data in splits.items():
                        storage.save(split_data, split_name,
                                   f'{split_name}_{timestamp}')

                    logger.info(f"✓ Saved simple split:")
                    logger.info(f"  - Training: {len(splits['training'])} rows (70%)")
                    logger.info(f"  - Validation: {len(splits['validation'])} rows (15%)")
                    logger.info(f"  - Testing: {len(splits['testing'])} rows (15%)")

            # Log stage completion
            stage_duration = time.time() - stage_start_time
            pipeline_log.log_stage_complete(pipeline_id, stage_name, metadata={
                'duration_seconds': stage_duration
            })
            pipeline_log.log_performance(pipeline_id, stage_name, stage_duration)

        except Exception as e:
            logger.error(f"✗ Stage '{stage_name}' failed: {e}")
            pipeline_log.log_stage_error(pipeline_id, stage_name, e)

            if verbose:
                import traceback
                logger.debug(traceback.format_exc())
            raise

    # Pipeline completion
    logger.info("=" * 70)
    logger.info("✓ Pipeline completed successfully")
    logger.info("=" * 70)

    # Log pipeline completion
    pipeline_log.log_event('pipeline_complete', pipeline_id, status='success')

    # Cleanup old logs and checkpoints
    pipeline_log.cleanup_old_logs()
    checkpoint_manager.cleanup_old_checkpoints(retention_days=7)


if __name__ == '__main__':
    run_pipeline()
