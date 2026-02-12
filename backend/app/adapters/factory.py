"""Factory for creating adapter instances based on configuration."""

import logging
from typing import Optional

from .base import CVProviderAdapter, LLMProviderAdapter
from .yolo_adapter import YOLOAdapter
from .easyocr_adapter import EasyOCRAdapter
from .llava_adapter import LLaVAAdapter
from ..config import settings

logger = logging.getLogger(__name__)


class AdapterFactory:
    """Factory for creating CV and LLM adapters."""
    
    _cv_adapters: dict[str, CVProviderAdapter] = {}
    _llm_adapters: dict[str, LLMProviderAdapter] = {}
    
    @classmethod
    def get_cv_adapter(cls, provider: Optional[str] = None) -> CVProviderAdapter:
        """Get CV adapter instance.
        
        Args:
            provider: Provider name (yolo, easyocr). If None, uses config default.
            
        Returns:
            CV adapter instance
        """
        provider = provider or settings.cv_provider
        provider = provider.lower()
        
        # Return cached instance if exists
        if provider in cls._cv_adapters:
            return cls._cv_adapters[provider]
        
        # Create new instance
        if provider == "yolo":
            adapter = YOLOAdapter()
        elif provider == "easyocr":
            adapter = EasyOCRAdapter()
        else:
            raise ValueError(f"Unknown CV provider: {provider}")
        
        cls._cv_adapters[provider] = adapter
        logger.info(f"Created CV adapter: {provider}")
        
        return adapter
    
    @classmethod
    def get_llm_adapter(cls, provider: Optional[str] = None) -> LLMProviderAdapter:
        """Get LLM adapter instance.
        
        Args:
            provider: Provider name (llava). If None, uses config default.
            
        Returns:
            LLM adapter instance
        """
        provider = provider or settings.llm_provider
        provider = provider.lower()
        
        # Return cached instance if exists
        if provider in cls._llm_adapters:
            return cls._llm_adapters[provider]
        
        # Create new instance
        if provider == "llava":
            adapter = LLaVAAdapter()
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
        
        cls._llm_adapters[provider] = adapter
        logger.info(f"Created LLM adapter: {provider}")
        
        return adapter
    
    @classmethod
    def cleanup_all(cls) -> None:
        """Cleanup all adapter instances."""
        logger.info("Cleaning up all adapters")
        
        for provider, adapter in cls._cv_adapters.items():
            try:
                adapter.unload_model()
            except Exception as e:
                logger.error(f"Error unloading CV adapter {provider}: {e}")
        
        for provider, adapter in cls._llm_adapters.items():
            try:
                adapter.unload_model()
            except Exception as e:
                logger.error(f"Error unloading LLM adapter {provider}: {e}")
        
        cls._cv_adapters.clear()
        cls._llm_adapters.clear()
        
        logger.info("All adapters cleaned up")


# Convenience functions
def get_yolo_adapter() -> YOLOAdapter:
    """Get YOLO adapter instance."""
    return AdapterFactory.get_cv_adapter("yolo")


def get_ocr_adapter() -> EasyOCRAdapter:
    """Get EasyOCR adapter instance."""
    return AdapterFactory.get_cv_adapter("easyocr")


def get_llava_adapter() -> LLaVAAdapter:
    """Get LLaVA adapter instance."""
    return AdapterFactory.get_llm_adapter("llava")
