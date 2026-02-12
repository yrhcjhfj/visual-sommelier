"""YOLO adapter for object detection using YOLOv8."""

import logging
from typing import List, Optional
from io import BytesIO
import numpy as np
from PIL import Image

from .base import CVProviderAdapter, Label, TextAnnotation, DetectedObject
from ..utils.gpu_utils import gpu_manager
from ..config import settings

logger = logging.getLogger(__name__)


class YOLOAdapter(CVProviderAdapter):
    """YOLOv8 implementation of CV provider."""
    
    def __init__(self):
        self._model = None
        self._device = None
        self._model_path = settings.yolo_model
        self._confidence_threshold = settings.yolo_confidence
        
    def load_model(self) -> None:
        """Load YOLOv8 model into GPU/CPU."""
        if self._model is not None:
            logger.debug("YOLO model already loaded")
            return
        
        try:
            from ultralytics import YOLO
            
            self._device = gpu_manager.get_device()
            logger.info(f"Loading YOLO model on {self._device}")
            
            # Load model
            self._model = YOLO(self._model_path)
            
            # Move to device
            if self._device == "cuda":
                self._model.to(self._device)
                if settings.use_mixed_precision:
                    # Enable FP16 for faster inference
                    self._model.model.half()
                    logger.info("YOLO model using FP16 precision")
            
            logger.info(f"YOLO model loaded successfully on {self._device}")
            gpu_manager.log_gpu_stats()
            
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise
    
    def unload_model(self) -> None:
        """Unload YOLO model from memory."""
        if self._model is not None:
            try:
                del self._model
                self._model = None
                gpu_manager.clear_gpu_cache()
                logger.info("YOLO model unloaded")
                gpu_manager.log_gpu_stats()
            except Exception as e:
                logger.error(f"Error unloading YOLO model: {e}")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
    
    def _bytes_to_image(self, image_bytes: bytes) -> Image.Image:
        """Convert bytes to PIL Image."""
        return Image.open(BytesIO(image_bytes))
    
    def detect_labels(self, image_bytes: bytes) -> List[Label]:
        """Detect objects and return labels."""
        if not self.is_loaded():
            self.load_model()
        
        try:
            image = self._bytes_to_image(image_bytes)
            
            # Run inference
            results = self._model(image, verbose=False)
            
            # Extract unique labels
            labels = []
            seen_classes = set()
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = result.names[class_id]
                        
                        if confidence >= self._confidence_threshold and class_name not in seen_classes:
                            labels.append(Label(
                                name=class_name,
                                confidence=confidence
                            ))
                            seen_classes.add(class_name)
            
            # Sort by confidence
            labels.sort(key=lambda x: x.confidence, reverse=True)
            logger.debug(f"Detected {len(labels)} unique labels")
            
            return labels
            
        except Exception as e:
            logger.error(f"Error detecting labels: {e}")
            return []
    
    def detect_text(self, image_bytes: bytes) -> List[TextAnnotation]:
        """YOLO doesn't do OCR, return empty list."""
        logger.debug("YOLO adapter doesn't support text detection")
        return []
    
    def detect_objects(self, image_bytes: bytes) -> List[DetectedObject]:
        """Detect objects with bounding boxes."""
        if not self.is_loaded():
            self.load_model()
        
        try:
            image = self._bytes_to_image(image_bytes)
            img_width, img_height = image.size
            
            # Run inference
            results = self._model(image, verbose=False)
            
            # Extract objects
            objects = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = result.names[class_id]
                        
                        if confidence >= self._confidence_threshold:
                            # Get bounding box in xyxy format
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = xyxy
                            
                            # Convert to (x, y, width, height) normalized
                            x = float(x1 / img_width)
                            y = float(y1 / img_height)
                            width = float((x2 - x1) / img_width)
                            height = float((y2 - y1) / img_height)
                            
                            objects.append(DetectedObject(
                                class_name=class_name,
                                confidence=confidence,
                                bounding_box=(x, y, width, height)
                            ))
            
            # Sort by confidence
            objects.sort(key=lambda x: x.confidence, reverse=True)
            logger.debug(f"Detected {len(objects)} objects")
            
            return objects
            
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return []
