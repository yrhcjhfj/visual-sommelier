# Adapters and GPU Utilities

This document describes the adapter system and GPU utilities implemented for the Visual Sommelier backend.

## Overview

The adapter system provides a modular architecture for integrating Computer Vision (CV) and Large Language Model (LLM) providers. This allows easy switching between different implementations without changing business logic.

## Components

### GPU Utilities (`app/utils/gpu_utils.py`)

Provides GPU detection, memory monitoring, and management capabilities.

**Key Features:**
- CUDA availability detection
- GPU memory monitoring
- GPU statistics (utilization, temperature)
- Automatic fallback to CPU if GPU unavailable
- GPU cache management
- NVML integration for detailed stats

**Usage:**
```python
from app.utils.gpu_utils import gpu_manager

# Check CUDA availability
if gpu_manager.check_cuda_availability():
    print("CUDA is available")

# Get device string
device = gpu_manager.get_device()  # Returns "cuda" or "cpu"

# Get GPU memory info
memory_info = gpu_manager.get_gpu_memory_info()
print(f"Used: {memory_info['used_mb']} MB")

# Get comprehensive GPU stats
stats = gpu_manager.get_gpu_stats()
if stats:
    print(f"Device: {stats.device_name}")
    print(f"Memory: {stats.used_memory_mb}/{stats.total_memory_mb} MB")
    print(f"Utilization: {stats.utilization_percent}%")

# Log GPU stats
gpu_manager.log_gpu_stats()

# Clear GPU cache
gpu_manager.clear_gpu_cache()

# Cleanup on shutdown
gpu_manager.cleanup()
```

### Base Adapters (`app/adapters/base.py`)

Defines abstract interfaces for CV and LLM providers.

**CVProviderAdapter Interface:**
- `detect_labels(image_bytes)` - Detect object labels
- `detect_text(image_bytes)` - Recognize text (OCR)
- `detect_objects(image_bytes)` - Detect objects with bounding boxes
- `load_model()` - Load model into memory/GPU
- `unload_model()` - Unload model from memory/GPU
- `is_loaded()` - Check if model is loaded

**LLMProviderAdapter Interface:**
- `generate_completion(prompt, image, language, max_tokens)` - Generate text
- `generate_structured_output(prompt, schema, language)` - Generate structured JSON
- `load_model()` - Load model into memory/GPU
- `unload_model()` - Unload model from memory/GPU
- `is_loaded()` - Check if model is loaded

### YOLO Adapter (`app/adapters/yolo_adapter.py`)

YOLOv8 implementation for object detection.

**Features:**
- Object detection with bounding boxes
- Label extraction
- GPU acceleration with CUDA
- FP16 mixed precision support
- Automatic model loading/unloading

**Usage:**
```python
from app.adapters import get_yolo_adapter

yolo = get_yolo_adapter()

# Detect objects
with open("image.jpg", "rb") as f:
    image_bytes = f.read()

objects = yolo.detect_objects(image_bytes)
for obj in objects:
    print(f"{obj.class_name}: {obj.confidence:.2f}")
    print(f"  Box: {obj.bounding_box}")

# Get labels
labels = yolo.detect_labels(image_bytes)
for label in labels:
    print(f"{label.name}: {label.confidence:.2f}")
```

### EasyOCR Adapter (`app/adapters/easyocr_adapter.py`)

EasyOCR implementation for text recognition.

**Features:**
- Multi-language OCR (English, Russian, Chinese)
- Text detection with bounding boxes
- GPU acceleration
- Automatic model loading/unloading

**Usage:**
```python
from app.adapters import get_ocr_adapter

ocr = get_ocr_adapter()

# Detect text
with open("image.jpg", "rb") as f:
    image_bytes = f.read()

texts = ocr.detect_text(image_bytes)
for text in texts:
    print(f"Text: {text.text}")
    print(f"  Confidence: {text.confidence:.2f}")
    print(f"  Box: {text.bounding_box}")
```

### LLaVA Adapter (`app/adapters/llava_adapter.py`)

LLaVA vision-language model via Ollama.

**Features:**
- Vision-language understanding
- Multi-language support (English, Russian, Chinese)
- Structured output generation
- 4-bit quantization for memory efficiency

**Usage:**
```python
from app.adapters import get_llava_adapter

llava = get_llava_adapter()

# Generate explanation
with open("device.jpg", "rb") as f:
    image_bytes = f.read()

explanation = llava.generate_completion(
    prompt="What is this device and how do I use it?",
    image=image_bytes,
    language="en"
)
print(explanation)

# Generate structured output
schema = {
    "device_type": "string",
    "confidence": "number",
    "description": "string"
}

result = llava.generate_structured_output(
    prompt="Identify this device",
    schema=schema,
    language="en"
)
print(result)
```

### Adapter Factory (`app/adapters/factory.py`)

Factory for creating and managing adapter instances.

**Features:**
- Singleton pattern for adapter instances
- Easy provider switching via configuration
- Centralized cleanup

**Usage:**
```python
from app.adapters.factory import AdapterFactory

# Get adapters by provider name
yolo = AdapterFactory.get_cv_adapter("yolo")
ocr = AdapterFactory.get_cv_adapter("easyocr")
llava = AdapterFactory.get_llm_adapter("llava")

# Or use convenience functions
from app.adapters import get_yolo_adapter, get_ocr_adapter, get_llava_adapter

yolo = get_yolo_adapter()
ocr = get_ocr_adapter()
llava = get_llava_adapter()

# Cleanup all adapters on shutdown
AdapterFactory.cleanup_all()
```

## Configuration

Adapters are configured via environment variables in `.env`:

```env
# GPU Configuration
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST=6.1

# Provider Selection
CV_PROVIDER=yolo
LLM_PROVIDER=llava

# YOLO Configuration
YOLO_MODEL=yolov8n.pt
YOLO_CONFIDENCE=0.5

# EasyOCR Configuration
EASYOCR_LANGUAGES=en,ru,zh

# LLaVA Configuration
OLLAMA_HOST=http://localhost:11434
LLAVA_MODEL=llava:7b-v1.6-mistral-q4_0
LLAVA_MAX_TOKENS=512
LLAVA_TEMPERATURE=0.7

# Model Management
MODEL_UNLOAD_TIMEOUT=300
ENABLE_MODEL_CACHING=True
USE_MIXED_PRECISION=True

# Logging
LOG_LEVEL=INFO
LOG_GPU_STATS=True
```

## FastAPI Integration

The adapters are integrated into the FastAPI application with lifecycle management:

```python
from app.utils.gpu_utils import gpu_manager
from app.adapters.factory import AdapterFactory

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    gpu_manager.check_cuda_availability()
    gpu_manager.log_gpu_stats()
    
    yield
    
    # Shutdown
    AdapterFactory.cleanup_all()
    gpu_manager.cleanup()

app = FastAPI(lifespan=lifespan)
```

## Testing

### Manual Testing

Run the manual test script:

```bash
cd backend
python test_adapters_manual.py
```

This will:
1. Test GPU utilities
2. Test adapter initialization
3. Test model loading/unloading
4. Display GPU statistics

### Unit Tests

Run unit tests:

```bash
cd backend
pytest tests/test_gpu_utils.py -v
pytest tests/test_adapters.py -v
```

## Memory Management

The system implements efficient GPU memory management:

1. **Sequential Loading**: Models are loaded on-demand
2. **Automatic Unloading**: Models can be unloaded after use
3. **Cache Clearing**: GPU cache is cleared when unloading
4. **Mixed Precision**: FP16 for CV models, 4-bit for LLM
5. **Monitoring**: Continuous GPU memory monitoring

**Memory Usage:**
- YOLO: ~500MB VRAM
- EasyOCR: ~500MB VRAM
- LLaVA: ~2.5GB VRAM
- Peak: ~3.5GB VRAM (all models loaded)

## Requirements

The adapters require the following dependencies:

- `torch>=2.2.0` - PyTorch with CUDA support
- `ultralytics>=8.1.0` - YOLOv8
- `easyocr>=1.7.1` - EasyOCR
- `ollama>=0.1.6` - Ollama client
- `nvidia-ml-py3>=7.352.0` - NVIDIA Management Library
- `pillow>=10.2.0` - Image processing
- `opencv-python>=4.9.0` - Computer vision utilities

## Architecture Benefits

1. **Modularity**: Easy to add new providers
2. **Testability**: Each adapter can be tested independently
3. **Flexibility**: Switch providers via configuration
4. **Maintainability**: Clear separation of concerns
5. **Performance**: Efficient GPU memory management
6. **Reliability**: Automatic fallback to CPU

## Future Extensions

The adapter system can be extended with:

- Additional CV providers (CLIP, SAM, etc.)
- Additional LLM providers (Qwen-VL, GPT-4V, etc.)
- Model quantization options
- Batch processing optimizations
- Distributed inference support
