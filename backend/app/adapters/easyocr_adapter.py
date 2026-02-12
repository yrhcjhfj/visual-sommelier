"""EasyOCR adapter for text recognition."""

import logging
from typing import List
from io import BytesIO
import numpy as np
from PIL import Image

from .base import CVProviderAdapter, Label, TextAnnotation, DetectedObject
from ..utils.gpu_utils import gpu_manager
from ..config import settings

logger = logging.getLogger(__name__)


class EasyOCRAdapter(CVProviderAdapter):
    """EasyOCR implementation of CV provider for text recognition."""
    
    def __init__(self):
        self._reader = None
        self._device = None
        self._languages = settings.easyocr_languages.split(",")
        
    def load_model(self) -> None:
        """Load EasyOCR model into GPU/CPU."""
        if self._reader is not None:
            logger.debug("EasyOCR model already loaded")
            return
        
        try:
            import easyocr
            
            # Determine device
            use_gpu = gpu_manager.check_cuda_availability()
            self._device = "cuda" if use_gpu else "cpu"
            
            logger.info(f"Loading EasyOCR model on {self._device} with languages: {self._languages}")
            
            # Load reader
            self._reader = easyocr.Reader(
                self._languages,
                gpu=use_gpu,
                verbose=False
            )
            
            logger.info(f"EasyOCR model loaded successfully on {self._device}")
            gpu_manager.log_gpu_stats()
            
        except Exception as e:
            logger.error(f"Error loading EasyOCR model: {e}")
            raise
    
    def unload_model(self) -> None:
        """Unload EasyOCR model from memory."""
        if self._reader is not None:
            try:
                del self._reader
                self._reader = None
                gpu_manager.clear_gpu_cache()
                logger.info("EasyOCR model unloaded")
                gpu_manager.log_gpu_stats()
            except Exception as e:
                logger.error(f"Error unloading EasyOCR model: {e}")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._reader is not None
    
    def _bytes_to_numpy(self, image_bytes: bytes) -> np.ndarray:
        """Convert bytes to numpy array."""
        image = Image.open(BytesIO(image_bytes))
        return np.array(image)
    
    def detect_labels(self, image_bytes: bytes) -> List[Label]:
        """EasyOCR doesn't detect object labels, return empty list."""
        logger.debug("EasyOCR adapter doesn't support label detection")
        return []
    
    def detect_text(self, image_bytes: bytes) -> List[TextAnnotation]:
        """Recognize text in the image."""
        if not self.is_loaded():
            self.load_model()
        
        try:
            image_array = self._bytes_to_numpy(image_bytes)
            img_height, img_width = image_array.shape[:2]
            
            # Run OCR
            results = self._reader.readtext(image_array)
            
            # Convert to TextAnnotation format
            annotations = []
            for detection in results:
                bbox, text, confidence = detection
                
                # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                # Convert to (x, y, width, height) normalized
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                x_min = min(x_coords)
                y_min = min(y_coords)
                x_max = max(x_coords)
                y_max = max(y_coords)
                
                x = x_min / img_width
                y = y_min / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                
                annotations.append(TextAnnotation(
                    text=text,
                    confidence=float(confidence),
                    bounding_box=(x, y, width, height)
                ))
            
            # Sort by confidence
            annotations.sort(key=lambda x: x.confidence, reverse=True)
            logger.debug(f"Detected {len(annotations)} text regions")
            
            return annotations
            
        except Exception as e:
            logger.error(f"Error detecting text: {e}")
            return []
    
    def detect_objects(self, image_bytes: bytes) -> List[DetectedObject]:
        """EasyOCR doesn't detect objects, return empty list."""
        logger.debug("EasyOCR adapter doesn't support object detection")
        return []
