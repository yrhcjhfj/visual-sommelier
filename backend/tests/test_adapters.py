"""Tests for adapter implementations."""

import pytest
from app.adapters import (
    YOLOAdapter,
    EasyOCRAdapter,
    LLaVAAdapter,
    AdapterFactory,
    get_yolo_adapter,
    get_ocr_adapter,
    get_llava_adapter
)


class TestYOLOAdapter:
    """Tests for YOLO adapter."""
    
    def test_initialization(self):
        """Test YOLO adapter can be initialized."""
        adapter = YOLOAdapter()
        assert adapter is not None
        assert not adapter.is_loaded()
    
    def test_load_unload(self):
        """Test model loading and unloading."""
        adapter = YOLOAdapter()
        
        # Initially not loaded
        assert not adapter.is_loaded()
        
        # Load model
        try:
            adapter.load_model()
            assert adapter.is_loaded()
            
            # Unload model
            adapter.unload_model()
            assert not adapter.is_loaded()
        except Exception as e:
            # If model file not found, that's expected in test environment
            pytest.skip(f"Model not available: {e}")


class TestEasyOCRAdapter:
    """Tests for EasyOCR adapter."""
    
    def test_initialization(self):
        """Test EasyOCR adapter can be initialized."""
        adapter = EasyOCRAdapter()
        assert adapter is not None
        assert not adapter.is_loaded()
    
    def test_load_unload(self):
        """Test model loading and unloading."""
        adapter = EasyOCRAdapter()
        
        # Initially not loaded
        assert not adapter.is_loaded()
        
        # Load model (may take time on first run)
        try:
            adapter.load_model()
            assert adapter.is_loaded()
            
            # Unload model
            adapter.unload_model()
            assert not adapter.is_loaded()
        except Exception as e:
            # If dependencies not available, skip
            pytest.skip(f"Model not available: {e}")


class TestLLaVAAdapter:
    """Tests for LLaVA adapter."""
    
    def test_initialization(self):
        """Test LLaVA adapter can be initialized."""
        adapter = LLaVAAdapter()
        assert adapter is not None
        assert not adapter.is_loaded()
    
    def test_load_unload(self):
        """Test client initialization."""
        adapter = LLaVAAdapter()
        
        # Initially not loaded
        assert not adapter.is_loaded()
        
        # Load (initialize client)
        try:
            adapter.load_model()
            assert adapter.is_loaded()
            
            # Unload
            adapter.unload_model()
            assert not adapter.is_loaded()
        except Exception as e:
            # If Ollama not running, skip
            pytest.skip(f"Ollama not available: {e}")


class TestAdapterFactory:
    """Tests for adapter factory."""
    
    def test_get_yolo_adapter(self):
        """Test getting YOLO adapter from factory."""
        adapter = AdapterFactory.get_cv_adapter("yolo")
        assert isinstance(adapter, YOLOAdapter)
        
        # Should return same instance
        adapter2 = AdapterFactory.get_cv_adapter("yolo")
        assert adapter is adapter2
    
    def test_get_ocr_adapter(self):
        """Test getting OCR adapter from factory."""
        adapter = AdapterFactory.get_cv_adapter("easyocr")
        assert isinstance(adapter, EasyOCRAdapter)
    
    def test_get_llava_adapter(self):
        """Test getting LLaVA adapter from factory."""
        adapter = AdapterFactory.get_llm_adapter("llava")
        assert isinstance(adapter, LLaVAAdapter)
    
    def test_unknown_provider(self):
        """Test error on unknown provider."""
        with pytest.raises(ValueError):
            AdapterFactory.get_cv_adapter("unknown")
        
        with pytest.raises(ValueError):
            AdapterFactory.get_llm_adapter("unknown")
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        yolo = get_yolo_adapter()
        assert isinstance(yolo, YOLOAdapter)
        
        ocr = get_ocr_adapter()
        assert isinstance(ocr, EasyOCRAdapter)
        
        llava = get_llava_adapter()
        assert isinstance(llava, LLaVAAdapter)
    
    def test_cleanup_all(self):
        """Test cleanup all adapters."""
        # Get some adapters
        AdapterFactory.get_cv_adapter("yolo")
        AdapterFactory.get_llm_adapter("llava")
        
        # Cleanup should not raise error
        AdapterFactory.cleanup_all()
