"""Tests for GPU utilities."""

import pytest
from app.utils.gpu_utils import GPUManager


def test_gpu_manager_initialization():
    """Test GPU manager can be initialized."""
    manager = GPUManager()
    assert manager is not None


def test_check_cuda_availability():
    """Test CUDA availability check."""
    manager = GPUManager()
    # Should return bool without error
    result = manager.check_cuda_availability()
    assert isinstance(result, bool)


def test_get_device():
    """Test device string retrieval."""
    manager = GPUManager()
    device = manager.get_device()
    assert device in ["cuda", "cpu"]


def test_get_gpu_memory_info():
    """Test GPU memory info retrieval."""
    manager = GPUManager()
    memory_info = manager.get_gpu_memory_info()
    
    assert isinstance(memory_info, dict)
    assert "total_mb" in memory_info
    assert "used_mb" in memory_info
    assert "free_mb" in memory_info
    
    # Values should be non-negative
    assert memory_info["total_mb"] >= 0
    assert memory_info["used_mb"] >= 0
    assert memory_info["free_mb"] >= 0


def test_get_gpu_stats():
    """Test GPU stats retrieval."""
    manager = GPUManager()
    stats = manager.get_gpu_stats()
    
    # If CUDA available, should return stats
    if manager.check_cuda_availability():
        assert stats is not None
        assert stats.device_id == 0
        assert stats.total_memory_mb > 0
    else:
        # If no CUDA, should return None
        assert stats is None


def test_clear_gpu_cache():
    """Test GPU cache clearing."""
    manager = GPUManager()
    # Should not raise error
    manager.clear_gpu_cache()


def test_cleanup():
    """Test cleanup."""
    manager = GPUManager()
    # Should not raise error
    manager.cleanup()
