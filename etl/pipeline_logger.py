"""Pipeline Event Logger with Rotation and Structured Activity Tracking.

Mathematical Foundation:
- Event stream: E = {e₁, e₂, ..., eₙ} with timestamps t₁ < t₂ < ... < tₙ
- Retention policy: Keep events where t_now - t_i ≤ 7 days
- Log rotation: Size-based (10MB) and time-based (daily) rotation

Success Criteria:
- <5ms logging overhead per event
- Structured JSON format for analysis
- Automatic cleanup of old logs (7-day retention)
- Thread-safe operations
"""
import logging
import json
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import os
import glob

logger = logging.getLogger(__name__)


class PipelineLogger:
    """Manages structured event logging with rotation and cleanup."""

    def __init__(self,
                 log_dir: str = "logs",
                 retention_days: int = 7,
                 max_bytes: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        """Initialize pipeline logger with rotation.

        Args:
            log_dir: Directory for log files
            retention_days: Days to retain logs (default: 7)
            max_bytes: Max size per log file before rotation (default: 10MB)
            backup_count: Number of backup files to keep
        """
        self.log_dir = Path(log_dir)
        self.retention_days = retention_days
        self.max_bytes = max_bytes
        self.backup_count = backup_count

        # Create log directory structure
        self._ensure_directories()

        # Initialize loggers
        self.pipeline_logger = self._setup_pipeline_logger()
        self.event_logger = self._setup_event_logger()
        self.error_logger = self._setup_error_logger()

    def _ensure_directories(self) -> None:
        """Create log directory structure."""
        (self.log_dir).mkdir(parents=True, exist_ok=True)
        (self.log_dir / "stages").mkdir(exist_ok=True)
        (self.log_dir / "events").mkdir(exist_ok=True)
        (self.log_dir / "errors").mkdir(exist_ok=True)

    def _setup_pipeline_logger(self) -> logging.Logger:
        """Setup main pipeline logger with rotation."""
        logger = logging.getLogger('pipeline')
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        # Rotating file handler (size-based)
        log_file = self.log_dir / "pipeline.log"
        handler = RotatingFileHandler(
            log_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _setup_event_logger(self) -> logging.Logger:
        """Setup structured event logger with daily rotation."""
        logger = logging.getLogger('pipeline_events')
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        # Time-based rotating handler (daily)
        log_file = self.log_dir / "events" / "events.log"
        handler = TimedRotatingFileHandler(
            log_file,
            when='midnight',
            interval=1,
            backupCount=self.retention_days
        )

        # JSON formatter for structured logging
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _setup_error_logger(self) -> logging.Logger:
        """Setup error logger with rotation."""
        logger = logging.getLogger('pipeline_errors')
        logger.setLevel(logging.ERROR)
        logger.handlers.clear()

        # Rotating file handler for errors
        log_file = self.log_dir / "errors" / "errors.log"
        handler = RotatingFileHandler(
            log_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s\n'
            'Exception: %(exc_info)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def log_event(self,
                  event_type: str,
                  pipeline_id: str,
                  stage: Optional[str] = None,
                  status: str = "success",
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log structured pipeline event.

        Args:
            event_type: Type of event (e.g., 'stage_start', 'stage_complete', 'checkpoint_saved')
            pipeline_id: Unique pipeline execution ID
            stage: Pipeline stage name
            status: Event status (success, error, warning)
            metadata: Additional event metadata
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'pipeline_id': pipeline_id,
            'stage': stage,
            'status': status,
            'metadata': metadata or {}
        }

        # Log as JSON for structured analysis
        self.event_logger.info(json.dumps(event))

        # Also log to main pipeline logger for visibility
        log_msg = f"[{event_type}] Pipeline: {pipeline_id}"
        if stage:
            log_msg += f", Stage: {stage}"
        log_msg += f", Status: {status}"

        if status == "success":
            self.pipeline_logger.info(log_msg)
        elif status == "warning":
            self.pipeline_logger.warning(log_msg)
        elif status == "error":
            self.pipeline_logger.error(log_msg)

    def log_stage_start(self, pipeline_id: str, stage: str, metadata: Optional[Dict] = None) -> None:
        """Log pipeline stage start."""
        self.log_event(
            event_type='stage_start',
            pipeline_id=pipeline_id,
            stage=stage,
            status='info',
            metadata=metadata
        )

    def log_stage_complete(self, pipeline_id: str, stage: str, metadata: Optional[Dict] = None) -> None:
        """Log pipeline stage completion."""
        self.log_event(
            event_type='stage_complete',
            pipeline_id=pipeline_id,
            stage=stage,
            status='success',
            metadata=metadata
        )

    def log_stage_error(self, pipeline_id: str, stage: str, error: Exception, metadata: Optional[Dict] = None) -> None:
        """Log pipeline stage error."""
        error_metadata = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            **(metadata or {})
        }

        self.log_event(
            event_type='stage_error',
            pipeline_id=pipeline_id,
            stage=stage,
            status='error',
            metadata=error_metadata
        )

        # Also log to error logger with full exception info
        self.error_logger.error(
            f"Stage '{stage}' failed in pipeline '{pipeline_id}': {str(error)}",
            exc_info=True
        )

    def log_checkpoint(self, pipeline_id: str, stage: str, checkpoint_id: str, metadata: Optional[Dict] = None) -> None:
        """Log checkpoint creation."""
        checkpoint_metadata = {
            'checkpoint_id': checkpoint_id,
            **(metadata or {})
        }

        self.log_event(
            event_type='checkpoint_saved',
            pipeline_id=pipeline_id,
            stage=stage,
            status='success',
            metadata=checkpoint_metadata
        )

    def log_data_quality(self, pipeline_id: str, stage: str, metrics: Dict[str, Any]) -> None:
        """Log data quality metrics."""
        self.log_event(
            event_type='data_quality_check',
            pipeline_id=pipeline_id,
            stage=stage,
            status='success',
            metadata={'metrics': metrics}
        )

    def log_performance(self, pipeline_id: str, stage: str, duration_seconds: float, metadata: Optional[Dict] = None) -> None:
        """Log performance metrics."""
        perf_metadata = {
            'duration_seconds': duration_seconds,
            **(metadata or {})
        }

        self.log_event(
            event_type='performance_metric',
            pipeline_id=pipeline_id,
            stage=stage,
            status='success',
            metadata=perf_metadata
        )

    def cleanup_old_logs(self) -> int:
        """Remove log files older than retention period.

        Returns:
            Number of log files deleted
        """
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        deleted_count = 0

        # Find all log files recursively
        log_patterns = [
            str(self.log_dir / "**" / "*.log"),
            str(self.log_dir / "**" / "*.log.*")
        ]

        for pattern in log_patterns:
            for log_file in glob.glob(pattern, recursive=True):
                log_path = Path(log_file)

                # Skip current active log files
                if log_path.name in ['pipeline.log', 'events.log', 'errors.log']:
                    continue

                # Check file modification time
                mtime = datetime.fromtimestamp(log_path.stat().st_mtime)

                if mtime < cutoff:
                    try:
                        log_path.unlink()
                        deleted_count += 1
                        logger.info(f"Deleted old log file: {log_file}")
                    except Exception as e:
                        logger.warning(f"Failed to delete log file {log_file}: {e}")

        if deleted_count > 0:
            logger.info(f"✓ Cleaned {deleted_count} old log file(s)")

        return deleted_count

    def get_recent_events(self,
                         pipeline_id: Optional[str] = None,
                         event_type: Optional[str] = None,
                         hours: int = 24) -> List[Dict[str, Any]]:
        """Retrieve recent events from logs.

        Args:
            pipeline_id: Filter by pipeline ID
            event_type: Filter by event type
            hours: Number of hours to look back

        Returns:
            List of event dictionaries
        """
        events = []
        cutoff = datetime.now() - timedelta(hours=hours)

        # Read from current and recent event logs
        event_log = self.log_dir / "events" / "events.log"

        if not event_log.exists():
            return events

        try:
            with open(event_log, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        event_time = datetime.fromisoformat(event['timestamp'])

                        # Apply filters
                        if event_time < cutoff:
                            continue

                        if pipeline_id and event.get('pipeline_id') != pipeline_id:
                            continue

                        if event_type and event.get('event_type') != event_type:
                            continue

                        events.append(event)

                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue

        except Exception as e:
            logger.warning(f"Error reading event log: {e}")

        return events

    def get_pipeline_summary(self, pipeline_id: str) -> Dict[str, Any]:
        """Get summary of pipeline execution events.

        Args:
            pipeline_id: Pipeline execution ID

        Returns:
            Dictionary with pipeline execution summary
        """
        events = self.get_recent_events(pipeline_id=pipeline_id, hours=24*7)  # Last 7 days

        if not events:
            return {
                'pipeline_id': pipeline_id,
                'total_events': 0,
                'stages_completed': [],
                'errors': [],
                'warnings': []
            }

        stages_completed = []
        errors = []
        warnings = []

        for event in events:
            if event['event_type'] == 'stage_complete':
                stages_completed.append(event['stage'])
            elif event['status'] == 'error':
                errors.append({
                    'stage': event.get('stage'),
                    'timestamp': event['timestamp'],
                    'error': event.get('metadata', {}).get('error_message')
                })
            elif event['status'] == 'warning':
                warnings.append({
                    'stage': event.get('stage'),
                    'timestamp': event['timestamp']
                })

        return {
            'pipeline_id': pipeline_id,
            'total_events': len(events),
            'stages_completed': stages_completed,
            'errors': errors,
            'warnings': warnings,
            'first_event': events[0]['timestamp'] if events else None,
            'last_event': events[-1]['timestamp'] if events else None
        }
