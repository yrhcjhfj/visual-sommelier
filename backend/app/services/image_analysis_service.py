"""Image analysis service for device identification and control detection."""

import logging
from typing import List, Optional
from io import BytesIO
from PIL import Image, ImageDraw
import uuid

from ..models.device import DeviceAnalysisResult, Control, BoundingBox
from ..adapters.factory import AdapterFactory
from ..adapters.base import DetectedObject, TextAnnotation

logger = logging.getLogger(__name__)


class ImageAnalysisService:
    """Service for analyzing device images and detecting controls."""
    
    # Confidence threshold for device identification
    HIGH_CONFIDENCE_THRESHOLD = 0.7
    LOW_CONFIDENCE_THRESHOLD = 0.3
    
    # Device categories for suggestions
    DEVICE_CATEGORIES = [
        "washing_machine",
        "remote_control",
        "microwave",
        "oven",
        "dishwasher",
        "air_conditioner",
        "tv",
        "furniture",
        "coffee_machine",
        "vacuum_cleaner",
        "other_appliance"
    ]
    
    # Control element keywords
    CONTROL_KEYWORDS = [
        "button", "knob", "switch", "lever", "dial",
        "slider", "touchpad", "display", "screen"
    ]

    def __init__(self):
        """Initialize the image analysis service."""
        self.adapter_factory = AdapterFactory()
        self._yolo_adapter = None
        self._ocr_adapter = None
    
    def _get_yolo_adapter(self):
        """Get or create YOLO adapter."""
        if self._yolo_adapter is None:
            self._yolo_adapter = self.adapter_factory.get_cv_adapter("yolo")
        return self._yolo_adapter
    
    def _get_ocr_adapter(self):
        """Get or create OCR adapter."""
        if self._ocr_adapter is None:
            self._ocr_adapter = self.adapter_factory.get_cv_adapter("easyocr")
        return self._ocr_adapter
    
    def _map_object_to_device_type(self, objects: List[DetectedObject]) -> tuple[str, float]:
        """Map detected objects to device type with confidence.
        
        Args:
            objects: List of detected objects
            
        Returns:
            Tuple of (device_type, confidence)
        """
        if not objects:
            return ("unknown", 0.0)
        
        # Get the highest confidence object
        top_object = objects[0]
        class_name = top_object.class_name.lower()
        confidence = top_object.confidence

        # Map common YOLO classes to device types
        device_mapping = {
            "microwave": "microwave",
            "oven": "oven",
            "refrigerator": "refrigerator",
            "toaster": "toaster",
            "remote": "remote_control",
            "tv": "tv",
            "laptop": "laptop",
            "keyboard": "keyboard",
            "mouse": "mouse",
            "cell phone": "cell_phone",
            "clock": "clock",
            "bottle": "bottle",
            "cup": "cup",
            "chair": "furniture",
            "couch": "furniture",
            "bed": "furniture",
        }
        
        # Try to find a match
        for key, device_type in device_mapping.items():
            if key in class_name:
                return (device_type, confidence)
        
        # If no specific match, use the class name as device type
        return (class_name.replace(" ", "_"), confidence)

    def analyze_device(self, image_bytes: bytes) -> DeviceAnalysisResult:
        """Analyze device image and identify the device type.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            DeviceAnalysisResult with device type, confidence, and suggestions
        """
        logger.info("Starting device analysis")
        
        try:
            # Get YOLO adapter and detect objects
            yolo = self._get_yolo_adapter()
            objects = yolo.detect_objects(image_bytes)
            
            # Map objects to device type
            device_type, confidence = self._map_object_to_device_type(objects)
            
            # Determine if we need to suggest categories
            suggested_categories = []
            if confidence < self.HIGH_CONFIDENCE_THRESHOLD:
                # Low confidence - suggest categories
                suggested_categories = self.DEVICE_CATEGORIES.copy()
                logger.info(f"Low confidence ({confidence:.2f}), suggesting categories")
            
            # Detect controls on the device
            detected_controls = self.detect_controls(image_bytes, device_type)
            
            result = DeviceAnalysisResult(
                device_type=device_type,
                confidence=confidence,
                brand=None,  # Brand detection not implemented yet
                model=None,  # Model detection not implemented yet
                suggested_categories=suggested_categories,
                detected_controls=detected_controls
            )
            
            logger.info(f"Device analysis complete: {device_type} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing device: {e}")
            # Return a default result with low confidence
            return DeviceAnalysisResult(
                device_type="unknown",
                confidence=0.0,
                suggested_categories=self.DEVICE_CATEGORIES.copy(),
                detected_controls=[]
            )

    def detect_controls(self, image_bytes: bytes, device_type: str) -> List[Control]:
        """Detect control elements on the device.
        
        Args:
            image_bytes: Image data as bytes
            device_type: Type of device (for context)
            
        Returns:
            List of detected Control objects
        """
        logger.info(f"Detecting controls for device type: {device_type}")
        
        try:
            controls = []
            
            # Get YOLO adapter and detect objects
            yolo = self._get_yolo_adapter()
            objects = yolo.detect_objects(image_bytes)
            
            # Filter objects that might be controls
            for obj in objects:
                class_name = obj.class_name.lower()
                
                # Check if this object could be a control element
                is_control = any(keyword in class_name for keyword in self.CONTROL_KEYWORDS)
                
                if is_control or obj.confidence > 0.6:
                    # Create a control from this object
                    control_id = str(uuid.uuid4())
                    control_type = self._infer_control_type(class_name)
                    
                    control = Control(
                        id=control_id,
                        type=control_type,
                        label=None,  # Will be filled by OCR
                        bounding_box=BoundingBox(
                            x=obj.bounding_box[0],
                            y=obj.bounding_box[1],
                            width=obj.bounding_box[2],
                            height=obj.bounding_box[3]
                        ),
                        confidence=obj.confidence
                    )
                    controls.append(control)

            # Get OCR adapter and detect text
            ocr = self._get_ocr_adapter()
            text_annotations = ocr.detect_text(image_bytes)
            
            # Try to match text to controls based on proximity
            for text_ann in text_annotations:
                matched = False
                text_bbox = text_ann.bounding_box
                
                # Find the closest control to this text
                for control in controls:
                    if self._is_text_near_control(text_bbox, control.bounding_box):
                        # Update control label
                        if control.label is None:
                            control.label = text_ann.text
                        else:
                            control.label += f" {text_ann.text}"
                        matched = True
                        break
                
                # If text doesn't match any control, create a new control for it
                if not matched and text_ann.confidence > 0.5:
                    control_id = str(uuid.uuid4())
                    control = Control(
                        id=control_id,
                        type="labeled_element",
                        label=text_ann.text,
                        bounding_box=BoundingBox(
                            x=text_bbox[0],
                            y=text_bbox[1],
                            width=text_bbox[2],
                            height=text_bbox[3]
                        ),
                        confidence=text_ann.confidence
                    )
                    controls.append(control)
            
            logger.info(f"Detected {len(controls)} controls")
            return controls
            
        except Exception as e:
            logger.error(f"Error detecting controls: {e}")
            return []

    def _infer_control_type(self, class_name: str) -> str:
        """Infer control type from class name.
        
        Args:
            class_name: Detected class name
            
        Returns:
            Control type string
        """
        class_name_lower = class_name.lower()
        
        if "button" in class_name_lower:
            return "button"
        elif "knob" in class_name_lower or "dial" in class_name_lower:
            return "knob"
        elif "switch" in class_name_lower:
            return "switch"
        elif "lever" in class_name_lower:
            return "lever"
        elif "slider" in class_name_lower:
            return "slider"
        elif "screen" in class_name_lower or "display" in class_name_lower:
            return "display"
        else:
            return "unknown_control"
    
    def _is_text_near_control(
        self,
        text_bbox: tuple[float, float, float, float],
        control_bbox: BoundingBox,
        threshold: float = 0.1
    ) -> bool:
        """Check if text is near a control element.
        
        Args:
            text_bbox: Text bounding box (x, y, width, height)
            control_bbox: Control bounding box
            threshold: Distance threshold (normalized)
            
        Returns:
            True if text is near control
        """
        # Calculate centers
        text_center_x = text_bbox[0] + text_bbox[2] / 2
        text_center_y = text_bbox[1] + text_bbox[3] / 2
        
        control_center_x = control_bbox.x + control_bbox.width / 2
        control_center_y = control_bbox.y + control_bbox.height / 2
        
        # Calculate distance
        distance = ((text_center_x - control_center_x) ** 2 + 
                   (text_center_y - control_center_y) ** 2) ** 0.5
        
        return distance < threshold

    def highlight_area(
        self,
        image_bytes: bytes,
        coordinates: BoundingBox,
        color: str = "red",
        width: int = 3
    ) -> bytes:
        """Highlight a specific area on the image.
        
        Args:
            image_bytes: Original image data
            coordinates: Bounding box to highlight
            color: Color for the highlight (default: red)
            width: Line width for the highlight (default: 3)
            
        Returns:
            Modified image with highlighted area as bytes
        """
        logger.info(f"Highlighting area at ({coordinates.x}, {coordinates.y})")
        
        try:
            # Load image
            image = Image.open(BytesIO(image_bytes))
            img_width, img_height = image.size
            
            # Convert normalized coordinates to pixel coordinates
            x1 = int(coordinates.x * img_width)
            y1 = int(coordinates.y * img_height)
            x2 = int((coordinates.x + coordinates.width) * img_width)
            y2 = int((coordinates.y + coordinates.height) * img_height)
            
            # Draw rectangle
            draw = ImageDraw.Draw(image)
            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline=color,
                width=width
            )
            
            # Convert back to bytes
            output = BytesIO()
            image.save(output, format=image.format or "PNG")
            output.seek(0)
            
            logger.info("Area highlighted successfully")
            return output.read()
            
        except Exception as e:
            logger.error(f"Error highlighting area: {e}")
            # Return original image on error
            return image_bytes

    def identify_control_at_coordinates(
        self,
        image_bytes: bytes,
        x: float,
        y: float,
        device_type: Optional[str] = None
    ) -> Optional[Control]:
        """Identify a control element at specific coordinates.
        
        This method implements Requirement 2.4: When a user points to an area
        on the image, the system identifies the control element in that area.
        
        Args:
            image_bytes: Image data as bytes
            x: X coordinate (normalized 0.0-1.0)
            y: Y coordinate (normalized 0.0-1.0)
            device_type: Optional device type for context
            
        Returns:
            Control object if found at coordinates, None otherwise
        """
        logger.info(f"Identifying control at coordinates ({x:.3f}, {y:.3f})")
        
        # Validate coordinates
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            logger.warning(f"Invalid coordinates: ({x}, {y})")
            return None
        
        try:
            # Detect all controls on the image
            controls = self.detect_controls(image_bytes, device_type or "unknown")
            
            if not controls:
                logger.info("No controls detected on image")
                return None
            
            # Find control that contains the point
            for control in controls:
                bbox = control.bounding_box
                
                # Check if point is inside bounding box
                if (bbox.x <= x <= bbox.x + bbox.width and
                    bbox.y <= y <= bbox.y + bbox.height):
                    logger.info(f"Found control '{control.type}' at coordinates")
                    return control
            
            # If no exact match, find the closest control
            closest_control = None
            min_distance = float('inf')
            
            for control in controls:
                bbox = control.bounding_box
                # Calculate center of bounding box
                center_x = bbox.x + bbox.width / 2
                center_y = bbox.y + bbox.height / 2
                
                # Calculate distance from point to center
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                
                if distance < min_distance:
                    min_distance = distance
                    closest_control = control
            
            # Return closest control if within reasonable distance (0.15 normalized units)
            if closest_control and min_distance < 0.15:
                logger.info(f"Found closest control '{closest_control.type}' at distance {min_distance:.3f}")
                return closest_control
            
            logger.info("No control found near coordinates")
            return None
            
        except Exception as e:
            logger.error(f"Error identifying control at coordinates: {e}")
            return None

    def identify_control_at_coordinates(
        self,
        image_bytes: bytes,
        x: float,
        y: float,
        device_type: Optional[str] = None
    ) -> Optional[Control]:
        """Identify a control element at specific coordinates.

        This method implements Requirement 2.4: When a user points to an area
        on the image, the system identifies the control element in that area.

        Args:
            image_bytes: Image data as bytes
            x: X coordinate (normalized 0.0-1.0)
            y: Y coordinate (normalized 0.0-1.0)
            device_type: Optional device type for context

        Returns:
            Control object if found at coordinates, None otherwise
        """
        logger.info(f"Identifying control at coordinates ({x:.3f}, {y:.3f})")

        # Validate coordinates
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            logger.warning(f"Invalid coordinates: ({x}, {y})")
            return None

        try:
            # Detect all controls on the image
            controls = self.detect_controls(image_bytes, device_type or "unknown")

            if not controls:
                logger.info("No controls detected on image")
                return None

            # Find control that contains the point
            for control in controls:
                bbox = control.bounding_box

                # Check if point is inside bounding box
                if (bbox.x <= x <= bbox.x + bbox.width and
                    bbox.y <= y <= bbox.y + bbox.height):
                    logger.info(f"Found control '{control.type}' at coordinates")
                    return control

            # If no exact match, find the closest control
            closest_control = None
            min_distance = float('inf')

            for control in controls:
                bbox = control.bounding_box
                # Calculate center of bounding box
                center_x = bbox.x + bbox.width / 2
                center_y = bbox.y + bbox.height / 2

                # Calculate distance from point to center
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5

                if distance < min_distance:
                    min_distance = distance
                    closest_control = control

            # Return closest control if within reasonable distance (0.15 normalized units)
            if closest_control and min_distance < 0.15:
                logger.info(f"Found closest control '{closest_control.type}' at distance {min_distance:.3f}")
                return closest_control

            logger.info("No control found near coordinates")
            return None

        except Exception as e:
            logger.error(f"Error identifying control at coordinates: {e}")
            return None

