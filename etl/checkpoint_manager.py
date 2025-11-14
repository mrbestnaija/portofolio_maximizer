"""Checkpoint Manager for ETL Pipeline State Persistence.

Mathematical Foundation:
- State vector: S(t) = {stage, data_hash, metadata, timestamp}
- Atomic persistence: temp → rename pattern for consistency
- Recovery: S(t_failed) → S(t_last_valid)

Success Criteria:
- <50ms checkpoint save/load
- Automatic recovery on pipeline failure
- No state corruption on concurrent access
"""
import json
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages pipeline state persistence and recovery."""

    def __init__(self, checkpoint_dir: str = "data/checkpoints"):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for storing checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self._ensure_metadata()

    def _ensure_metadata(self) -> None:
        """Ensure metadata file exists."""
        if not self.metadata_file.exists():
            self._save_metadata({
                'checkpoints': {},
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            })

    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save metadata atomically."""
        temp_path = self.metadata_file.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        temp_path.replace(self.metadata_file)

    def _load_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata."""
        if not self.metadata_file.exists():
            return {'checkpoints': {}}

        with open(self.metadata_file, 'r') as f:
            return json.load(f)

    def _compute_data_hash(self, data: pd.DataFrame) -> str:
        """Compute stable hash for DataFrame.

        Args:
            data: DataFrame to hash

        Returns:
            SHA256 hash of data content
        """
        # Use stable representation: sorted index + values
        data_sorted = data.sort_index()
        hash_input = pd.util.hash_pandas_object(data_sorted, index=True).values
        return hashlib.sha256(hash_input.tobytes()).hexdigest()[:16]

    def save_checkpoint(self,
                       pipeline_id: str,
                       stage: str,
                       data: pd.DataFrame,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save pipeline checkpoint with state persistence.

        Args:
            pipeline_id: Unique pipeline execution ID
            stage: Pipeline stage name
            data: DataFrame to checkpoint
            metadata: Additional metadata to save

        Returns:
            Checkpoint ID
        """
        # Generate checkpoint ID
        timestamp = datetime.now()
        data_hash = self._compute_data_hash(data)
        checkpoint_id = f"{pipeline_id}_{stage}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        # Create checkpoint paths
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.parquet"
        state_path = self.checkpoint_dir / f"{checkpoint_id}_state.pkl"

        # Save data (atomic write)
        temp_data = checkpoint_path.with_suffix('.tmp')
        data.to_parquet(temp_data, compression='snappy', index=True)
        temp_data.rename(checkpoint_path)

        # Save state metadata
        state = {
            'checkpoint_id': checkpoint_id,
            'pipeline_id': pipeline_id,
            'stage': stage,
            'data_hash': data_hash,
            'data_shape': data.shape,
            'data_columns': list(data.columns),
            'index_type': str(type(data.index).__name__),
            'timestamp': timestamp.isoformat(),
            'metadata': metadata or {}
        }

        temp_state = state_path.with_suffix('.tmp')
        with open(temp_state, 'wb') as f:
            pickle.dump(state, f)
        temp_state.rename(state_path)

        # Update metadata registry
        meta = self._load_metadata()
        if pipeline_id not in meta['checkpoints']:
            meta['checkpoints'][pipeline_id] = []

        meta['checkpoints'][pipeline_id].append({
            'checkpoint_id': checkpoint_id,
            'stage': stage,
            'timestamp': timestamp.isoformat(),
            'data_shape': data.shape,
            'data_hash': data_hash
        })
        meta['last_updated'] = datetime.now().isoformat()
        self._save_metadata(meta)

        logger.info(f"✓ Checkpoint saved: {checkpoint_id} "
                   f"(stage={stage}, shape={data.shape}, hash={data_hash})")

        return checkpoint_id

    def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Load checkpoint data and state.

        Args:
            checkpoint_id: Checkpoint ID to load

        Returns:
            Dictionary with 'data' (DataFrame) and 'state' (metadata)

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.parquet"
        state_path = self.checkpoint_dir / f"{checkpoint_id}_state.pkl"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")

        # Load data
        data = pd.read_parquet(checkpoint_path)

        # Load state
        with open(state_path, 'rb') as f:
            state = pickle.load(f)

        # Verify data integrity
        loaded_hash = self._compute_data_hash(data)
        if loaded_hash != state['data_hash']:
            logger.warning(f"Data hash mismatch for {checkpoint_id}: "
                         f"expected {state['data_hash']}, got {loaded_hash}")

        logger.info(f"✓ Checkpoint loaded: {checkpoint_id} "
                   f"(stage={state['stage']}, shape={data.shape})")

        return {
            'data': data,
            'state': state
        }

    def get_latest_checkpoint(self, pipeline_id: str, stage: Optional[str] = None) -> Optional[str]:
        """Get the most recent checkpoint for a pipeline.

        Args:
            pipeline_id: Pipeline execution ID
            stage: Optional stage filter

        Returns:
            Checkpoint ID or None if not found
        """
        meta = self._load_metadata()
        checkpoints = meta.get('checkpoints', {}).get(pipeline_id, [])

        if not checkpoints:
            return None

        # Filter by stage if specified
        if stage:
            checkpoints = [cp for cp in checkpoints if cp['stage'] == stage]

        if not checkpoints:
            return None

        # Return most recent
        latest = max(checkpoints, key=lambda x: x['timestamp'])
        return latest['checkpoint_id']

    def list_checkpoints(self, pipeline_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all checkpoints, optionally filtered by pipeline_id.

        Args:
            pipeline_id: Optional pipeline ID filter

        Returns:
            List of checkpoint metadata dictionaries
        """
        meta = self._load_metadata()
        checkpoints = meta.get('checkpoints', {})

        if pipeline_id:
            return checkpoints.get(pipeline_id, [])

        # Return all checkpoints
        all_checkpoints = []
        for pid, cp_list in checkpoints.items():
            for cp in cp_list:
                cp['pipeline_id'] = pid
                all_checkpoints.append(cp)

        return sorted(all_checkpoints, key=lambda x: x['timestamp'], reverse=True)

    def cleanup_old_checkpoints(self, retention_days: int = 7) -> int:
        """Remove checkpoints older than retention period.

        Args:
            retention_days: Keep checkpoints modified within this many days

        Returns:
            Number of checkpoints deleted
        """
        cutoff = datetime.now() - timedelta(days=retention_days)
        meta = self._load_metadata()
        checkpoints = meta.get('checkpoints', {})

        deleted_count = 0
        updated_checkpoints = {}

        for pipeline_id, cp_list in checkpoints.items():
            valid_checkpoints = []

            for cp in cp_list:
                cp_time = datetime.fromisoformat(cp['timestamp'])

                if cp_time < cutoff:
                    # Delete checkpoint files
                    checkpoint_id = cp['checkpoint_id']
                    data_file = self.checkpoint_dir / f"{checkpoint_id}.parquet"
                    state_file = self.checkpoint_dir / f"{checkpoint_id}_state.pkl"

                    if data_file.exists():
                        data_file.unlink()
                    if state_file.exists():
                        state_file.unlink()

                    deleted_count += 1
                    logger.info(f"Deleted old checkpoint: {checkpoint_id}")
                else:
                    valid_checkpoints.append(cp)

            if valid_checkpoints:
                updated_checkpoints[pipeline_id] = valid_checkpoints

        # Update metadata
        meta['checkpoints'] = updated_checkpoints
        meta['last_updated'] = datetime.now().isoformat()
        self._save_metadata(meta)

        if deleted_count > 0:
            logger.info(f"✓ Cleaned {deleted_count} old checkpoint(s)")

        return deleted_count

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to delete

        Returns:
            True if deleted, False if not found
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.parquet"
        state_path = self.checkpoint_dir / f"{checkpoint_id}_state.pkl"

        if not checkpoint_path.exists():
            return False

        # Delete files
        checkpoint_path.unlink()
        if state_path.exists():
            state_path.unlink()

        # Update metadata
        meta = self._load_metadata()
        updated = False

        for pipeline_id in list(meta.get('checkpoints', {}).keys()):
            cp_list = meta['checkpoints'][pipeline_id]
            original_len = len(cp_list)
            meta['checkpoints'][pipeline_id] = [
                cp for cp in cp_list if cp['checkpoint_id'] != checkpoint_id
            ]
            if len(meta['checkpoints'][pipeline_id]) < original_len:
                updated = True

            # Remove empty pipeline entries
            if not meta['checkpoints'][pipeline_id]:
                del meta['checkpoints'][pipeline_id]

        if updated:
            meta['last_updated'] = datetime.now().isoformat()
            self._save_metadata(meta)
            logger.info(f"✓ Deleted checkpoint: {checkpoint_id}")

        return True

    def get_pipeline_progress(self, pipeline_id: str) -> Dict[str, Any]:
        """Get progress information for a pipeline execution.

        Args:
            pipeline_id: Pipeline execution ID

        Returns:
            Dictionary with progress information
        """
        meta = self._load_metadata()
        checkpoints = meta.get('checkpoints', {}).get(pipeline_id, [])

        if not checkpoints:
            return {
                'pipeline_id': pipeline_id,
                'completed_stages': [],
                'total_checkpoints': 0,
                'latest_checkpoint': None
            }

        stages = [cp['stage'] for cp in checkpoints]
        latest = max(checkpoints, key=lambda x: x['timestamp'])

        return {
            'pipeline_id': pipeline_id,
            'completed_stages': stages,
            'total_checkpoints': len(checkpoints),
            'latest_checkpoint': latest['checkpoint_id'],
            'latest_stage': latest['stage'],
            'latest_timestamp': latest['timestamp']
        }
