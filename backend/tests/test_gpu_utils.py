"""Tests for GPU utilities."""

import pytest
from app.utils.gpu_utils import GPUManager, GPUStats
from datetime import datetime


# ============================================================================
# CUDA Availability Tests
# ============================================================================

def test_gpu_manager_initialization():
    """Test GPU manager can be initialized."""
    manager = GPUManager()
    assert manager is not None
    assert manager._cuda_available is None  # Not checked yet
    assert manager._device_count == 0
    assert manager._device_name is None


def test_check_cuda_availability():
    """Test CUDA availability check returns boolean."""
    manager = GPUManager()
    result = manager.check_cuda_availability()
    assert isinstance(result, bool)


def test_check_cuda_availability_caching():
    """Test CUDA availability check is cached."""
    manager = GPUManager()
    first_result = manager.check_cuda_availability()
    second_result = manager.check_cuda_availability()
    
    # Should return same result without re-checking
    assert first_result == second_result
    assert manager._cuda_available is not None


def test_check_cuda_availability_sets_device_info():
    """Test CUDA check sets device information when available."""
    manager = GPUManager()
    is_available = manager.check_cuda_availability()
    
    if is_available:
        assert manager._device_count > 0
        assert manager._device_name is not None
        assert isinstance(manager._device_name, str)
    else:
        # On CPU-only systems
        assert manager._device_count == 0


def test_get_device():
    """Test device string retrieval."""
    manager = GPUManager()
    device = manager.get_device()
    assert device in ["cuda", "cpu"]
    
    # Should match CUDA availability
    cuda_available = manager.check_cuda_availability()
    if cuda_available:
        assert device == "cuda"
    else:
        assert device == "cpu"


# ============================================================================
# Memory Monitoring Tests
# ============================================================================

def test_get_gpu_memory_info_structure():
    """Test GPU memory info returns correct structure."""
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


def test_get_gpu_memory_info_when_cuda_available():
    """Test GPU memory info when CUDA is available."""
    manager = GPUManager()
    
    if manager.check_cuda_availability():
        memory_info = manager.get_gpu_memory_info()
        
        # Should have positive total memory
        assert memory_info["total_mb"] > 0
        
        # Used memory should be less than or equal to total
        assert memory_info["used_mb"] <= memory_info["total_mb"]
        
        # Free memory should be reasonable
        assert memory_info["free_mb"] >= 0
        
        # Should have reserved memory info
        assert "reserved_mb" in memory_info
        assert memory_info["reserved_mb"] >= 0


def test_get_gpu_memory_info_when_cuda_unavailable():
    """Test GPU memory info when CUDA is not available."""
    manager = GPUManager()
    
    if not manager.check_cuda_availability():
        memory_info = manager.get_gpu_memory_info()
        
        # Should return zeros
        assert memory_info["total_mb"] == 0
        assert memory_info["used_mb"] == 0
        assert memory_info["free_mb"] == 0


def test_get_gpu_stats_structure():
    """Test GPU stats returns correct structure."""
    manager = GPUManager()
    stats = manager.get_gpu_stats()
    
    if manager.check_cuda_availability():
        assert stats is not None
        assert isinstance(stats, GPUStats)
        assert stats.device_id == 0
        assert isinstance(stats.device_name, str)
        assert stats.total_memory_mb > 0
        assert stats.used_memory_mb >= 0
        assert stats.free_memory_mb >= 0
        assert stats.utilization_percent >= 0
        assert isinstance(stats.timestamp, datetime)
        
        # Temperature may be None if not available
        if stats.temperature_celsius is not None:
            assert stats.temperature_celsius > 0
    else:
        assert stats is None


def test_get_gpu_stats_memory_consistency():
    """Test GPU stats memory values are consistent."""
    manager = GPUManager()
    
    if manager.check_cuda_availability():
        stats = manager.get_gpu_stats()
        
        # Used + free should be close to total (accounting for rounding)
        # Note: free is calculated differently, so we just check reasonableness
        assert stats.used_memory_mb <= stats.total_memory_mb
        assert stats.free_memory_mb <= stats.total_memory_mb


def test_log_gpu_stats():
    """Test GPU stats logging doesn't raise errors."""
    manager = GPUManager()
    # Should not raise error regardless of CUDA availability
    manager.log_gpu_stats()


# ============================================================================
# Model Loading/Unloading Tests (Cache Management)
# ============================================================================

def test_clear_gpu_cache():
    """Test GPU cache clearing."""
    manager = GPUManager()
    # Should not raise error
    manager.clear_gpu_cache()


def test_clear_gpu_cache_when_cuda_available():
    """Test GPU cache clearing when CUDA is available."""
    manager = GPUManager()
    
    if manager.check_cuda_availability():
        # Get initial memory
        initial_memory = manager.get_gpu_memory_info()
        
        # Clear cache
        manager.clear_gpu_cache()
        
        # Get memory after clearing
        after_memory = manager.get_gpu_memory_info()
        
        # Memory info should still be valid
        assert after_memory["total_mb"] > 0
        assert after_memory["used_mb"] >= 0


def test_clear_gpu_cache_when_cuda_unavailable():
    """Test GPU cache clearing when CUDA is not available."""
    manager = GPUManager()
    
    if not manager.check_cuda_availability():
        # Should not raise error even without CUDA
        manager.clear_gpu_cache()


def test_cleanup():
    """Test cleanup."""
    manager = GPUManager()
    # Should not raise error
    manager.cleanup()


def test_cleanup_clears_cache():
    """Test cleanup clears GPU cache."""
    manager = GPUManager()
    
    if manager.check_cuda_availability():
        # Initialize NVML if possible
        manager._init_nvml()
        
        # Cleanup should clear cache and shutdown NVML
        manager.cleanup()
        
        # After cleanup, NVML should be shutdown
        if manager._pynvml is not None:
            assert manager._nvml_initialized == False


def test_multiple_cleanup_calls():
    """Test multiple cleanup calls don't cause errors."""
    manager = GPUManager()
    
    # Multiple cleanups should be safe
    manager.cleanup()
    manager.cleanup()
    manager.cleanup()


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_gpu_workflow():
    """Test complete GPU workflow: check -> monitor -> cleanup."""
    manager = GPUManager()
    
    # Check CUDA
    is_available = manager.check_cuda_availability()
    assert isinstance(is_available, bool)
    
    # Get device
    device = manager.get_device()
    assert device in ["cuda", "cpu"]
    
    # Monitor memory
    memory_info = manager.get_gpu_memory_info()
    assert isinstance(memory_info, dict)
    
    # Get stats
    stats = manager.get_gpu_stats()
    if is_available:
        assert stats is not None
    else:
        assert stats is None
    
    # Log stats
    manager.log_gpu_stats()
    
    # Clear cache
    manager.clear_gpu_cache()
    
    # Cleanup
    manager.cleanup()


def test_memory_monitoring_multiple_calls():
    """Test memory monitoring can be called multiple times."""
    manager = GPUManager()
    
    if manager.check_cuda_availability():
        # Multiple calls should work
        mem1 = manager.get_gpu_memory_info()
        mem2 = manager.get_gpu_memory_info()
        mem3 = manager.get_gpu_memory_info()
        
        # All should return valid data
        assert mem1["total_mb"] > 0
        assert mem2["total_mb"] > 0
        assert mem3["total_mb"] > 0
        
        # Total memory should be consistent
        assert mem1["total_mb"] == mem2["total_mb"] == mem3["total_mb"]
