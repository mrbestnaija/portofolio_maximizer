"""Unit tests for checkpoint_manager.py - State Persistence and Recovery.

Test Coverage:
- Checkpoint save/load operations
- Data integrity validation
- Metadata management
- Cleanup operations
- Pipeline progress tracking
- Error handling and edge cases
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import shutil
import tempfile

from etl.checkpoint_manager import CheckpointManager


class TestCheckpointManager:
    """Test suite for CheckpointManager."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary checkpoint directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame for testing."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'close': np.random.randn(100) * 10 + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        return data

    @pytest.fixture
    def checkpoint_manager(self, temp_checkpoint_dir):
        """Initialize CheckpointManager with temp directory."""
        return CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

    # ==================== Initialization Tests ====================

    def test_initialization_creates_directory(self, temp_checkpoint_dir):
        """Test that CheckpointManager creates checkpoint directory."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
        assert Path(temp_checkpoint_dir).exists()
        assert Path(temp_checkpoint_dir).is_dir()

    def test_initialization_creates_metadata_file(self, checkpoint_manager, temp_checkpoint_dir):
        """Test that metadata file is created on initialization."""
        metadata_file = Path(temp_checkpoint_dir) / "checkpoint_metadata.json"
        assert metadata_file.exists()

    def test_metadata_structure(self, checkpoint_manager):
        """Test metadata file has correct structure."""
        metadata = checkpoint_manager._load_metadata()
        assert 'checkpoints' in metadata
        assert 'created_at' in metadata
        assert 'last_updated' in metadata
        assert isinstance(metadata['checkpoints'], dict)

    # ==================== Save Checkpoint Tests ====================

    def test_save_checkpoint_basic(self, checkpoint_manager, sample_data):
        """Test basic checkpoint saving."""
        checkpoint_id = checkpoint_manager.save_checkpoint(
            pipeline_id='test_pipeline',
            stage='data_extraction',
            data=sample_data
        )

        assert checkpoint_id is not None
        assert 'test_pipeline' in checkpoint_id
        assert 'data_extraction' in checkpoint_id

    def test_save_checkpoint_creates_files(self, checkpoint_manager, sample_data, temp_checkpoint_dir):
        """Test that checkpoint files are created."""
        checkpoint_id = checkpoint_manager.save_checkpoint(
            pipeline_id='test_pipeline',
            stage='preprocessing',
            data=sample_data
        )

        # Check data file exists
        data_file = Path(temp_checkpoint_dir) / f"{checkpoint_id}.parquet"
        assert data_file.exists()

        # Check state file exists
        state_file = Path(temp_checkpoint_dir) / f"{checkpoint_id}_state.pkl"
        assert state_file.exists()

    def test_save_checkpoint_with_metadata(self, checkpoint_manager, sample_data):
        """Test checkpoint saving with additional metadata."""
        metadata = {
            'source': 'yfinance',
            'tickers': ['AAPL', 'MSFT'],
            'config': {'cache_hours': 24}
        }

        checkpoint_id = checkpoint_manager.save_checkpoint(
            pipeline_id='test_pipeline',
            stage='extraction',
            data=sample_data,
            metadata=metadata
        )

        # Load and verify metadata
        loaded = checkpoint_manager.load_checkpoint(checkpoint_id)
        assert loaded['state']['metadata'] == metadata

    def test_save_checkpoint_updates_registry(self, checkpoint_manager, sample_data):
        """Test that checkpoint updates metadata registry."""
        checkpoint_manager.save_checkpoint(
            pipeline_id='pipeline_1',
            stage='stage_1',
            data=sample_data
        )

        metadata = checkpoint_manager._load_metadata()
        assert 'pipeline_1' in metadata['checkpoints']
        assert len(metadata['checkpoints']['pipeline_1']) == 1

    def test_save_multiple_checkpoints_same_pipeline(self, checkpoint_manager, sample_data):
        """Test saving multiple checkpoints for same pipeline."""
        checkpoint_manager.save_checkpoint('pipe1', 'stage1', sample_data)
        checkpoint_manager.save_checkpoint('pipe1', 'stage2', sample_data)
        checkpoint_manager.save_checkpoint('pipe1', 'stage3', sample_data)

        metadata = checkpoint_manager._load_metadata()
        assert len(metadata['checkpoints']['pipe1']) == 3

    # ==================== Load Checkpoint Tests ====================

    def test_load_checkpoint_basic(self, checkpoint_manager, sample_data):
        """Test basic checkpoint loading."""
        checkpoint_id = checkpoint_manager.save_checkpoint(
            pipeline_id='test_pipeline',
            stage='test_stage',
            data=sample_data
        )

        loaded = checkpoint_manager.load_checkpoint(checkpoint_id)
        assert 'data' in loaded
        assert 'state' in loaded
        assert isinstance(loaded['data'], pd.DataFrame)

    def test_load_checkpoint_data_integrity(self, checkpoint_manager, sample_data):
        """Test that loaded data matches saved data."""
        checkpoint_id = checkpoint_manager.save_checkpoint(
            pipeline_id='test_pipeline',
            stage='test_stage',
            data=sample_data
        )

        loaded = checkpoint_manager.load_checkpoint(checkpoint_id)
        # Check data equality (frequencies may differ after parquet save/load)
        pd.testing.assert_frame_equal(loaded['data'], sample_data, check_freq=False)

    def test_load_checkpoint_state_content(self, checkpoint_manager, sample_data):
        """Test that checkpoint state contains expected fields."""
        checkpoint_id = checkpoint_manager.save_checkpoint(
            pipeline_id='test_pipeline',
            stage='test_stage',
            data=sample_data
        )

        loaded = checkpoint_manager.load_checkpoint(checkpoint_id)
        state = loaded['state']

        assert state['checkpoint_id'] == checkpoint_id
        assert state['pipeline_id'] == 'test_pipeline'
        assert state['stage'] == 'test_stage'
        assert state['data_shape'] == sample_data.shape
        assert set(state['data_columns']) == set(sample_data.columns)
        assert 'data_hash' in state
        assert 'timestamp' in state

    def test_load_nonexistent_checkpoint_raises_error(self, checkpoint_manager):
        """Test loading nonexistent checkpoint raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            checkpoint_manager.load_checkpoint('nonexistent_checkpoint_id')

    # ==================== Data Hash Tests ====================

    def test_data_hash_consistency(self, checkpoint_manager, sample_data):
        """Test that same data produces same hash."""
        hash1 = checkpoint_manager._compute_data_hash(sample_data)
        hash2 = checkpoint_manager._compute_data_hash(sample_data)
        assert hash1 == hash2

    def test_data_hash_different_for_different_data(self, checkpoint_manager):
        """Test that different data produces different hashes."""
        data1 = pd.DataFrame({'a': [1, 2, 3]})
        data2 = pd.DataFrame({'a': [4, 5, 6]})

        hash1 = checkpoint_manager._compute_data_hash(data1)
        hash2 = checkpoint_manager._compute_data_hash(data2)
        assert hash1 != hash2

    def test_data_hash_order_independent(self, checkpoint_manager):
        """Test that data hash is independent of row order (after sorting)."""
        data1 = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        }, index=[0, 1, 2])

        data2 = pd.DataFrame({
            'a': [3, 1, 2],
            'b': [6, 4, 5]
        }, index=[2, 0, 1])

        # After sorting by index, should produce same hash
        hash1 = checkpoint_manager._compute_data_hash(data1.sort_index())
        hash2 = checkpoint_manager._compute_data_hash(data2.sort_index())
        assert hash1 == hash2

    # ==================== Latest Checkpoint Tests ====================

    def test_get_latest_checkpoint(self, checkpoint_manager, sample_data):
        """Test retrieving latest checkpoint for a pipeline."""
        cp1 = checkpoint_manager.save_checkpoint('pipe1', 'stage1', sample_data)
        cp2 = checkpoint_manager.save_checkpoint('pipe1', 'stage2', sample_data)
        cp3 = checkpoint_manager.save_checkpoint('pipe1', 'stage3', sample_data)

        latest = checkpoint_manager.get_latest_checkpoint('pipe1')
        assert latest == cp3

    def test_get_latest_checkpoint_by_stage(self, checkpoint_manager, sample_data):
        """Test retrieving latest checkpoint filtered by stage."""
        checkpoint_manager.save_checkpoint('pipe1', 'extraction', sample_data)
        cp2 = checkpoint_manager.save_checkpoint('pipe1', 'preprocessing', sample_data)
        checkpoint_manager.save_checkpoint('pipe1', 'extraction', sample_data)

        latest_preprocessing = checkpoint_manager.get_latest_checkpoint('pipe1', stage='preprocessing')
        assert latest_preprocessing == cp2

    def test_get_latest_checkpoint_nonexistent_pipeline(self, checkpoint_manager):
        """Test getting latest checkpoint for nonexistent pipeline returns None."""
        latest = checkpoint_manager.get_latest_checkpoint('nonexistent_pipeline')
        assert latest is None

    # ==================== List Checkpoints Tests ====================

    def test_list_checkpoints_all(self, checkpoint_manager, sample_data):
        """Test listing all checkpoints."""
        checkpoint_manager.save_checkpoint('pipe1', 'stage1', sample_data)
        checkpoint_manager.save_checkpoint('pipe2', 'stage1', sample_data)
        checkpoint_manager.save_checkpoint('pipe1', 'stage2', sample_data)

        all_checkpoints = checkpoint_manager.list_checkpoints()
        assert len(all_checkpoints) == 3

    def test_list_checkpoints_filtered(self, checkpoint_manager, sample_data):
        """Test listing checkpoints filtered by pipeline_id."""
        checkpoint_manager.save_checkpoint('pipe1', 'stage1', sample_data)
        checkpoint_manager.save_checkpoint('pipe2', 'stage1', sample_data)
        checkpoint_manager.save_checkpoint('pipe1', 'stage2', sample_data)

        pipe1_checkpoints = checkpoint_manager.list_checkpoints('pipe1')
        assert len(pipe1_checkpoints) == 2

    def test_list_checkpoints_sorted_by_time(self, checkpoint_manager, sample_data):
        """Test that checkpoints are sorted by timestamp (newest first)."""
        checkpoint_manager.save_checkpoint('pipe1', 'stage1', sample_data)
        checkpoint_manager.save_checkpoint('pipe1', 'stage2', sample_data)
        checkpoint_manager.save_checkpoint('pipe1', 'stage3', sample_data)

        checkpoints = checkpoint_manager.list_checkpoints()
        timestamps = [cp['timestamp'] for cp in checkpoints]
        assert timestamps == sorted(timestamps, reverse=True)

    # ==================== Cleanup Tests ====================

    def test_cleanup_old_checkpoints(self, checkpoint_manager, sample_data, temp_checkpoint_dir):
        """Test cleanup of old checkpoints."""
        # Save checkpoint
        checkpoint_id = checkpoint_manager.save_checkpoint('pipe1', 'stage1', sample_data)

        # Manually modify timestamp to make it old
        metadata = checkpoint_manager._load_metadata()
        old_timestamp = (datetime.now() - timedelta(days=10)).isoformat()
        metadata['checkpoints']['pipe1'][0]['timestamp'] = old_timestamp
        checkpoint_manager._save_metadata(metadata)

        # Run cleanup (7-day retention)
        deleted_count = checkpoint_manager.cleanup_old_checkpoints(retention_days=7)

        assert deleted_count == 1

        # Verify files are deleted
        data_file = Path(temp_checkpoint_dir) / f"{checkpoint_id}.parquet"
        state_file = Path(temp_checkpoint_dir) / f"{checkpoint_id}_state.pkl"
        assert not data_file.exists()
        assert not state_file.exists()

    def test_cleanup_keeps_recent_checkpoints(self, checkpoint_manager, sample_data):
        """Test that recent checkpoints are kept during cleanup."""
        checkpoint_manager.save_checkpoint('pipe1', 'stage1', sample_data)
        checkpoint_manager.save_checkpoint('pipe1', 'stage2', sample_data)

        deleted_count = checkpoint_manager.cleanup_old_checkpoints(retention_days=7)

        assert deleted_count == 0
        assert len(checkpoint_manager.list_checkpoints('pipe1')) == 2

    def test_cleanup_mixed_old_and_recent(self, checkpoint_manager, sample_data):
        """Test cleanup with mix of old and recent checkpoints."""
        # Save recent checkpoint
        checkpoint_manager.save_checkpoint('pipe1', 'stage1', sample_data)

        # Save and age another checkpoint
        checkpoint_manager.save_checkpoint('pipe1', 'stage2', sample_data)
        metadata = checkpoint_manager._load_metadata()
        old_timestamp = (datetime.now() - timedelta(days=10)).isoformat()
        metadata['checkpoints']['pipe1'][1]['timestamp'] = old_timestamp
        checkpoint_manager._save_metadata(metadata)

        # Run cleanup
        deleted_count = checkpoint_manager.cleanup_old_checkpoints(retention_days=7)

        assert deleted_count == 1
        assert len(checkpoint_manager.list_checkpoints('pipe1')) == 1

    # ==================== Delete Checkpoint Tests ====================

    def test_delete_checkpoint(self, checkpoint_manager, sample_data, temp_checkpoint_dir):
        """Test deleting specific checkpoint."""
        checkpoint_id = checkpoint_manager.save_checkpoint('pipe1', 'stage1', sample_data)

        result = checkpoint_manager.delete_checkpoint(checkpoint_id)
        assert result is True

        # Verify files are deleted
        data_file = Path(temp_checkpoint_dir) / f"{checkpoint_id}.parquet"
        state_file = Path(temp_checkpoint_dir) / f"{checkpoint_id}_state.pkl"
        assert not data_file.exists()
        assert not state_file.exists()

    def test_delete_nonexistent_checkpoint(self, checkpoint_manager):
        """Test deleting nonexistent checkpoint returns False."""
        result = checkpoint_manager.delete_checkpoint('nonexistent_id')
        assert result is False

    def test_delete_checkpoint_updates_metadata(self, checkpoint_manager, sample_data):
        """Test that delete updates metadata registry."""
        cp1 = checkpoint_manager.save_checkpoint('pipe1', 'stage1', sample_data)
        checkpoint_manager.save_checkpoint('pipe1', 'stage2', sample_data)

        checkpoint_manager.delete_checkpoint(cp1)

        checkpoints = checkpoint_manager.list_checkpoints('pipe1')
        assert len(checkpoints) == 1
        assert checkpoints[0]['stage'] == 'stage2'

    # ==================== Pipeline Progress Tests ====================

    def test_get_pipeline_progress(self, checkpoint_manager, sample_data):
        """Test getting pipeline progress information."""
        checkpoint_manager.save_checkpoint('pipe1', 'extraction', sample_data)
        checkpoint_manager.save_checkpoint('pipe1', 'preprocessing', sample_data)
        checkpoint_manager.save_checkpoint('pipe1', 'storage', sample_data)

        progress = checkpoint_manager.get_pipeline_progress('pipe1')

        assert progress['pipeline_id'] == 'pipe1'
        assert len(progress['completed_stages']) == 3
        assert progress['total_checkpoints'] == 3
        assert progress['latest_stage'] == 'storage'
        assert progress['latest_checkpoint'] is not None

    def test_get_pipeline_progress_no_checkpoints(self, checkpoint_manager):
        """Test getting progress for pipeline with no checkpoints."""
        progress = checkpoint_manager.get_pipeline_progress('nonexistent_pipe')

        assert progress['pipeline_id'] == 'nonexistent_pipe'
        assert progress['completed_stages'] == []
        assert progress['total_checkpoints'] == 0
        assert progress['latest_checkpoint'] is None

    # ==================== Edge Cases and Error Handling ====================

    def test_atomic_write_prevents_corruption(self, checkpoint_manager, sample_data):
        """Test that atomic write pattern prevents corruption."""
        # Save checkpoint normally
        checkpoint_id = checkpoint_manager.save_checkpoint('pipe1', 'stage1', sample_data)

        # Verify no .tmp files remain
        temp_files = list(Path(checkpoint_manager.checkpoint_dir).glob("*.tmp"))
        assert len(temp_files) == 0

    def test_empty_dataframe_checkpoint(self, checkpoint_manager):
        """Test checkpointing empty DataFrame."""
        empty_df = pd.DataFrame()

        checkpoint_id = checkpoint_manager.save_checkpoint('pipe1', 'stage1', empty_df)
        loaded = checkpoint_manager.load_checkpoint(checkpoint_id)

        assert loaded['data'].empty
        assert loaded['state']['data_shape'] == (0, 0)

    def test_large_dataframe_checkpoint(self, checkpoint_manager):
        """Test checkpointing large DataFrame."""
        large_data = pd.DataFrame(
            np.random.randn(10000, 50),
            columns=[f'col_{i}' for i in range(50)]
        )

        checkpoint_id = checkpoint_manager.save_checkpoint('pipe1', 'stage1', large_data)
        loaded = checkpoint_manager.load_checkpoint(checkpoint_id)

        assert loaded['data'].shape == large_data.shape
        pd.testing.assert_frame_equal(loaded['data'], large_data)

    def test_checkpoint_with_multiindex(self, checkpoint_manager):
        """Test checkpointing DataFrame with MultiIndex."""
        index = pd.MultiIndex.from_tuples([
            ('A', 1), ('A', 2), ('B', 1), ('B', 2)
        ], names=['letter', 'number'])
        data = pd.DataFrame({'value': [10, 20, 30, 40]}, index=index)

        checkpoint_id = checkpoint_manager.save_checkpoint('pipe1', 'stage1', data)
        loaded = checkpoint_manager.load_checkpoint(checkpoint_id)

        pd.testing.assert_frame_equal(loaded['data'], data)
