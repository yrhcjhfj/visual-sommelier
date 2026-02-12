"""Manual test script for adapters and GPU utilities.

Run this script to verify that adapters are working correctly.
"""

import sys
import logging
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.utils.gpu_utils import gpu_manager
from app.adapters.factory import AdapterFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_gpu_utilities():
    """Test GPU utilities."""
    logger.info("=" * 60)
    logger.info("Testing GPU Utilities")
    logger.info("=" * 60)
    
    # Check CUDA availability
    cuda_available = gpu_manager.check_cuda_availability()
    logger.info(f"CUDA Available: {cuda_available}")
    
    # Get device
    device = gpu_manager.get_device()
    logger.info(f"Device: {device}")
    
    # Get memory info
    memory_info = gpu_manager.get_gpu_memory_info()
    logger.info(f"GPU Memory Info: {memory_info}")
    
    # Get GPU stats
    stats = gpu_manager.get_gpu_stats()
    if stats:
        logger.info(f"GPU Stats:")
        logger.info(f"  Device: {stats.device_name}")
        logger.info(f"  Memory: {stats.used_memory_mb:.0f}/{stats.total_memory_mb:.0f} MB")
        logger.info(f"  Utilization: {stats.utilization_percent:.1f}%")
        if stats.temperature_celsius:
            logger.info(f"  Temperature: {stats.temperature_celsius:.0f}°C")
    else:
        logger.info("GPU stats not available (running on CPU)")
    
    logger.info("")


def test_adapter_initialization():
    """Test adapter initialization."""
    logger.info("=" * 60)
    logger.info("Testing Adapter Initialization")
    logger.info("=" * 60)
    
    # Test YOLO adapter
    logger.info("Initializing YOLO adapter...")
    try:
        yolo = AdapterFactory.get_cv_adapter("yolo")
        logger.info(f"YOLO adapter created: {yolo.__class__.__name__}")
        logger.info(f"YOLO loaded: {yolo.is_loaded()}")
    except Exception as e:
        logger.error(f"Error with YOLO adapter: {e}")
    
    logger.info("")
    
    # Test EasyOCR adapter
    logger.info("Initializing EasyOCR adapter...")
    try:
        ocr = AdapterFactory.get_cv_adapter("easyocr")
        logger.info(f"EasyOCR adapter created: {ocr.__class__.__name__}")
        logger.info(f"EasyOCR loaded: {ocr.is_loaded()}")
    except Exception as e:
        logger.error(f"Error with EasyOCR adapter: {e}")
    
    logger.info("")
    
    # Test LLaVA adapter
    logger.info("Initializing LLaVA adapter...")
    try:
        llava = AdapterFactory.get_llm_adapter("llava")
        logger.info(f"LLaVA adapter created: {llava.__class__.__name__}")
        logger.info(f"LLaVA loaded: {llava.is_loaded()}")
    except Exception as e:
        logger.error(f"Error with LLaVA adapter: {e}")
    
    logger.info("")


def test_model_loading():
    """Test model loading and unloading."""
    logger.info("=" * 60)
    logger.info("Testing Model Loading/Unloading")
    logger.info("=" * 60)
    
    # Test YOLO model loading
    logger.info("Testing YOLO model loading...")
    try:
        yolo = AdapterFactory.get_cv_adapter("yolo")
        logger.info(f"Before load - Is loaded: {yolo.is_loaded()}")
        
        logger.info("Loading YOLO model...")
        yolo.load_model()
        logger.info(f"After load - Is loaded: {yolo.is_loaded()}")
        
        # Log GPU stats after loading
        gpu_manager.log_gpu_stats()
        
        logger.info("Unloading YOLO model...")
        yolo.unload_model()
        logger.info(f"After unload - Is loaded: {yolo.is_loaded()}")
        
        # Log GPU stats after unloading
        gpu_manager.log_gpu_stats()
        
    except Exception as e:
        logger.error(f"Error testing YOLO model: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("")


def cleanup():
    """Cleanup all resources."""
    logger.info("=" * 60)
    logger.info("Cleaning Up")
    logger.info("=" * 60)
    
    AdapterFactory.cleanup_all()
    gpu_manager.cleanup()
    
    logger.info("Cleanup complete")


def main():
    """Run all tests."""
    try:
        test_gpu_utilities()
        test_adapter_initialization()
        test_model_loading()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
