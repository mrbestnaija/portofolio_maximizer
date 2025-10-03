#!/usr/bin/env python3
"""ETL Pipeline Orchestration Script."""
import sys
import yaml
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import click

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.yfinance_extractor import YFinanceExtractor
from etl.data_validator import DataValidator
from etl.preprocessor import Preprocessor
from etl.data_storage import DataStorage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@click.command()
@click.option('--config', default='workflows/etl_pipeline.yml', help='Path to pipeline configuration')
@click.option('--tickers', default='AAPL,MSFT', help='Comma-separated ticker symbols')
@click.option('--start', default='2020-01-01', help='Start date (YYYY-MM-DD)')
@click.option('--end', default='2024-01-01', help='End date (YYYY-MM-DD)')
def run_pipeline(config: str, tickers: str, start: str, end: str) -> None:
    """Execute ETL pipeline with stage-by-stage orchestration."""
    workflow = yaml.safe_load(open(config))
    stages = {s['name']: s for s in workflow['stages']}
    ticker_list = [t.strip() for t in tickers.split(',')]
    storage = DataStorage()

    for stage_name in tqdm(['data_extraction', 'data_validation', 'data_preprocessing', 'data_storage'], desc='Pipeline'):
        logging.info(f"Starting stage: {stage_name}")

        if stage_name == 'data_extraction':
            # Initialize extractor with caching enabled (24h cache validity)
            extractor = YFinanceExtractor(storage=storage, cache_hours=24)
            raw_data = extractor.extract_ohlcv(ticker_list, start, end)
            if raw_data is None or raw_data.empty:
                raise RuntimeError("Data extraction failed")
            # Data is auto-cached in extract_ohlcv, no duplicate save needed

        elif stage_name == 'data_validation':
            validator = DataValidator()
            report = validator.validate_ohlcv(raw_data)
            if not report['passed']:
                logging.error(f"Validation failed: {report}")

        elif stage_name == 'data_preprocessing':
            processor = Preprocessor()
            # Handle missing values
            filled = processor.handle_missing(raw_data)
            # Normalize (returns tuple: data, stats)
            normalized, stats = processor.normalize(filled)
            processed = normalized
            storage.save(processed, 'processed', f'processed_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

        elif stage_name == 'data_storage':
            splits = storage.train_validation_test_split(processed, train_ratio=0.7, val_ratio=0.15)
            for split_name, split_data in splits.items():
                storage.save(split_data, split_name, f'{split_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

    logging.info("Pipeline completed successfully")

if __name__ == '__main__':
    run_pipeline()
