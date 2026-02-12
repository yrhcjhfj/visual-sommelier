"""GPU utilities for CUDA detection, memory monitoring, and model management."""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class GPUStats:
    """GPU statistics snapshot."""
    device_id: int
    device_name: str
    total_memory_mb: float
    used_memory_mb: float
    free_memory_mb: float
    utilization_percent: float
    temperature_celsius: Optional[float]
    timestamp: datetime


class GPUManager:
    """Manages GPU detection, memory monitoring, and model loading."""
    
    def __init__(self):
        self._cuda_available: Optional[bool] = None
        self._device_count: int = 0
        self._device_name: Optional[str] = None
        self._torch = None
        self._pynvml = None
        self._nvml_initialized = False
        
    def check_cuda_availability(self) -> bool:
        """Check if CUDA is available and functional.
        
        Returns:
            bool: True if CUDA is available, False otherwise
        """
        if self._cuda_available is not None:
            return self._cuda_available
            
        try:
            import torch
            self._torch = torch
            self._cuda_available = torch.cuda.is_available()
            
            if self._cuda_available:
                self._device_count = torch.cuda.device_count()
                self._device_name = torch.cuda.get_device_name(0)
                logger.info(
                    f"CUDA available: {self._device_count} device(s) detected. "
                    f"Primary device: {self._device_name}"
                )
            else:
                logger.warning("CUDA not available. Will fallback to CPU.")
                
        except Exception as e:
            logger.error(f"Error checking CUDA availability: {e}")
            self._cuda_available = False
            
        return self._cuda_available
    
    def get_device(self) -> str:
        """Get the device string for model loading.
        
        Returns:
            str: 'cuda' if available, 'cpu' otherwise
        """
        if self.check_cuda_availability():
            return "cuda"
        return "cpu"
    
    def _init_nvml(self) -> bool:
        """Initialize NVIDIA Management Library for detailed GPU stats.
        
        Returns:
            bool: True if initialization successful
        """
        if self._nvml_initialized:
            return True
            
        try:
            import pynvml
            self._pynvml = pynvml
            pynvml.nvmlInit()
            self._nvml_initialized = True
            logger.info("NVML initialized successfully")
            return True
        except Exception as e:
            logger.warning(f"Could not initialize NVML: {e}")
            return False
    
    def get_gpu_memory_info(self, device_id: int = 0) -> Dict[str, float]:
        """Get GPU memory information.
        
        Args:
            device_id: GPU device ID (default: 0)
            
        Returns:
            Dict with total, used, and free memory in MB
        """
        if not self.check_cuda_availability():
            return {"total_mb": 0, "used_mb": 0, "free_mb": 0}
        
        try:
            if self._torch is None:
                import torch
                self._torch = torch
                
            # Get memory info from PyTorch
            total = self._torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 2)
            reserved = self._torch.cuda.memory_reserved(device_id) / (1024 ** 2)
            allocated = self._torch.cuda.memory_allocated(device_id) / (1024 ** 2)
            free = total - reserved
            
            return {
                "total_mb": round(total, 2),
                "used_mb": round(allocated, 2),
                "reserved_mb": round(reserved, 2),
                "free_mb": round(free, 2)
            }
        except Exception as e:
            logger.error(f"Error getting GPU memory info: {e}")
            return {"total_mb": 0, "used_mb": 0, "free_mb": 0}
    
    def get_gpu_stats(self, device_id: int = 0) -> Optional[GPUStats]:
        """Get comprehensive GPU statistics.
        
        Args:
            device_id: GPU device ID (default: 0)
            
        Returns:
            GPUStats object or None if unavailable
        """
        if not self.check_cuda_availability():
            return None
        
        try:
            memory_info = self.get_gpu_memory_info(device_id)
            utilization = 0.0
            temperature = None
            
            # Try to get utilization and temperature from NVML
            if self._init_nvml() and self._pynvml:
                try:
                    handle = self._pynvml.nvmlDeviceGetHandleByIndex(device_id)
                    util = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization = float(util.gpu)
                    
                    try:
                        temperature = float(self._pynvml.nvmlDeviceGetTemperature(
                            handle, self._pynvml.NVML_TEMPERATURE_GPU
                        ))
                    except:
                        pass
                except Exception as e:
                    logger.debug(f"Could not get NVML stats: {e}")
            
            return GPUStats(
                device_id=device_id,
                device_name=self._device_name or "Unknown",
                total_memory_mb=memory_info["total_mb"],
                used_memory_mb=memory_info["used_mb"],
                free_memory_mb=memory_info["free_mb"],
                utilization_percent=utilization,
                temperature_celsius=temperature,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error getting GPU stats: {e}")
            return None
    
    def log_gpu_stats(self, device_id: int = 0) -> None:
        """Log current GPU statistics.
        
        Args:
            device_id: GPU device ID (default: 0)
        """
        stats = self.get_gpu_stats(device_id)
        if stats:
            logger.info(
                f"GPU Stats - Device: {stats.device_name}, "
                f"Memory: {stats.used_memory_mb:.0f}/{stats.total_memory_mb:.0f} MB "
                f"({stats.used_memory_mb/stats.total_memory_mb*100:.1f}%), "
                f"Utilization: {stats.utilization_percent:.1f}%"
                + (f", Temp: {stats.temperature_celsius:.0f}°C" if stats.temperature_celsius else "")
            )
        else:
            logger.info("GPU stats unavailable (running on CPU)")
    
    def clear_gpu_cache(self) -> None:
        """Clear GPU cache to free memory."""
        if self.check_cuda_availability() and self._torch:
            try:
                self._torch.cuda.empty_cache()
                logger.info("GPU cache cleared")
            except Exception as e:
                logger.error(f"Error clearing GPU cache: {e}")
    
    def cleanup(self) -> None:
        """Cleanup GPU resources."""
        self.clear_gpu_cache()
        if self._nvml_initialized and self._pynvml:
            try:
                self._pynvml.nvmlShutdown()
                self._nvml_initialized = False
                logger.info("NVML shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down NVML: {e}")


# Global GPU manager instance
gpu_manager = GPUManager()
