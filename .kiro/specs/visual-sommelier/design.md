# Документ проектирования

## Обзор

«Визуальный Сомелье» — это приложение, которое использует компьютерное зрение и большие языковые модели для помощи пользователям в понимании сложных бытовых устройств. Система анализирует изображения устройств, идентифицирует элементы управления и предоставляет понятные объяснения на естественном языке.

Приложение будет реализовано как веб-приложение с адаптивным интерфейсом, работающее на мобильных устройствах и десктопах. Это обеспечит максимальную доступность без необходимости установки из магазинов приложений.

## Архитектура

Система построена на трехуровневой архитектуре с локальным GPU-ускорением:

### Системные требования

**GPU:**
- NVIDIA Quadro P1000 (4GB VRAM) с CUDA 11.8+
- Compute Capability 6.1+
- CUDA Toolkit и cuDNN установлены

**CPU/RAM:**
- Минимум 8GB RAM (16GB рекомендуется)
- 4+ ядра CPU для preprocessing

**Хранилище:**
- 20GB для моделей (YOLOv8, CLIP, LLaVA-7B quantized)
- SSD рекомендуется для быстрой загрузки моделей

### Уровень представления (Frontend)
- **Технология**: React с TypeScript
- **UI Framework**: Tailwind CSS для адаптивного дизайна
- **Камера**: WebRTC API для доступа к камере устройства
- **Состояние**: Zustand для управления состоянием приложения
- **Хранилище**: IndexedDB для локального хранения истории и кэша

### Уровень бизнес-логики (Backend)
- **Технология**: Python с FastAPI
- **Архитектурный паттерн**: Clean Architecture с разделением на слои
- **API**: RESTful API для синхронных операций
- **Валидация**: Pydantic для валидации данных

### Уровень интеграции (Local GPU Inference)
- **CV Provider**: Локальные модели на NVIDIA Quadro P1000 с CUDA
  - YOLOv8 для детекции объектов
  - CLIP для классификации устройств
  - EasyOCR для распознавания текста
- **LLM Provider**: Локальные vision-language модели
  - LLaVA (7B/13B) через Ollama
  - Qwen-VL как альтернатива
  - Поддержка квантизации (4-bit) для оптимизации памяти
- **Хранилище изображений**: Локальное хранилище

### Диаграмма архитектуры

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[React UI]
        Camera[Camera Module]
        Storage[IndexedDB Storage]
    end
    
    subgraph "Backend Layer"
        API[FastAPI Server]
        SessionMgr[Session Manager]
        ImageProc[Image Processor]
        TextGen[Text Generator]
    end
    
    subgraph "Local GPU Inference Layer - NVIDIA Quadro P1000"
        CVAdapter[CV Adapter Interface]
        LLMAdapter[LLM Adapter Interface]
        YOLO[YOLOv8 Object Detection]
        CLIP[CLIP Classification]
        OCR[EasyOCR Text Recognition]
        LLaVA[LLaVA Vision-Language Model]
        Qwen[Qwen-VL Alternative]
    end
    
    UI --> API
    Camera --> UI
    Storage --> UI
    
    API --> SessionMgr
    API --> ImageProc
    API --> TextGen
    
    ImageProc --> CVAdapter
    TextGen --> LLMAdapter
    
    CVAdapter --> YOLO
    CVAdapter --> CLIP
    CVAdapter --> OCR
    LLMAdapter --> LLaVA
    LLMAdapter --> Qwen
```

## Компоненты и интерфейсы

### Frontend компоненты

#### 1. CameraCapture
**Назначение**: Управление камерой и захват изображений

**Интерфейс**:
```typescript
interface CameraCapture {
  startCamera(): Promise<void>;
  stopCamera(): void;
  captureImage(): Promise<Blob>;
  switchCamera(): void; // Переключение между фронтальной и задней камерой
  isActive: boolean;
}
```

#### 2. ImageUploader
**Назначение**: Загрузка изображений из галереи

**Интерфейс**:
```typescript
interface ImageUploader {
  selectImage(): Promise<File>;
  validateImage(file: File): boolean;
  compressImage(file: File, maxSize: number): Promise<Blob>;
}
```

#### 3. DeviceIdentifier
**Назначение**: Отображение результатов идентификации устройства

**Интерфейс**:
```typescript
interface DeviceInfo {
  deviceType: string;
  confidence: number;
  suggestedCategories?: string[];
}

interface DeviceIdentifier {
  displayDeviceInfo(info: DeviceInfo): void;
  allowManualSelection(): Promise<string>;
}
```

#### 4. ExplanationView
**Назначение**: Отображение объяснений и инструкций

**Интерфейс**:
```typescript
interface Explanation {
  text: string;
  steps?: Step[];
  warnings?: string[];
  highlightedAreas?: BoundingBox[];
}

interface ExplanationView {
  displayExplanation(explanation: Explanation): void;
  markStepComplete(stepIndex: number): void;
  requestClarification(stepIndex: number): Promise<string>;
}
```

#### 5. HistoryManager
**Назначение**: Управление историей сессий

**Интерфейс**:
```typescript
interface Session {
  id: string;
  timestamp: Date;
  deviceImage: Blob;
  deviceType: string;
  conversation: Message[];
}

interface HistoryManager {
  saveSession(session: Session): Promise<void>;
  loadSessions(): Promise<Session[]>;
  deleteSession(id: string): Promise<void>;
  restoreSession(id: string): Promise<Session>;
}
```

### Backend компоненты

#### 1. ImageAnalysisService
**Назначение**: Координация анализа изображений

**Интерфейс**:
```python
class ImageAnalysisService:
    def analyze_device(self, image: bytes) -> DeviceAnalysisResult:
        """Анализирует изображение и идентифицирует устройство"""
        pass
    
    def detect_controls(self, image: bytes, device_type: str) -> List[Control]:
        """Обнаруживает элементы управления на изображении"""
        pass
    
    def highlight_area(self, image: bytes, coordinates: BoundingBox) -> bytes:
        """Выделяет область на изображении"""
        pass
```

#### 2. ExplanationService
**Назначение**: Генерация объяснений с использованием LLM

**Интерфейс**:
```python
class ExplanationService:
    def generate_explanation(
        self, 
        image: bytes, 
        question: str, 
        device_context: DeviceContext,
        language: str
    ) -> Explanation:
        """Генерирует объяснение на основе изображения и вопроса"""
        pass
    
    def generate_instructions(
        self, 
        task: str, 
        device_context: DeviceContext,
        language: str
    ) -> List[Step]:
        """Генерирует пошаговые инструкции для задачи"""
        pass
    
    def clarify_step(
        self, 
        step: Step, 
        question: str,
        language: str
    ) -> str:
        """Предоставляет дополнительные разъяснения для шага"""
        pass
```

#### 3. SessionService
**Назначение**: Управление сессиями пользователей

**Интерфейс**:
```python
class SessionService:
    def create_session(self, user_id: str, device_image: bytes) -> Session:
        """Создает новую сессию"""
        pass
    
    def add_message(self, session_id: str, message: Message) -> None:
        """Добавляет сообщение в сессию"""
        pass
    
    def get_session_context(self, session_id: str) -> SessionContext:
        """Получает контекст сессии для LLM"""
        pass
```

### Integration Layer - Адаптеры

#### 1. CVProviderAdapter (Абстрактный интерфейс)
**Назначение**: Единый интерфейс для провайдеров компьютерного зрения

**Интерфейс**:
```python
from abc import ABC, abstractmethod

class CVProviderAdapter(ABC):
    @abstractmethod
    def detect_labels(self, image: bytes) -> List[Label]:
        """Обнаруживает объекты и метки на изображении"""
        pass
    
    @abstractmethod
    def detect_text(self, image: bytes) -> List[TextAnnotation]:
        """Распознает текст на изображении"""
        pass
    
    @abstractmethod
    def detect_objects(self, image: bytes) -> List[DetectedObject]:
        """Обнаруживает объекты с координатами"""
        pass
```

**Реализации**:
- `YOLOAdapter`: Использует YOLOv8 для детекции объектов и элементов управления
- `CLIPAdapter`: Использует CLIP для классификации типов устройств
- `EasyOCRAdapter`: Использует EasyOCR для распознавания текста на устройствах

#### 2. LLMProviderAdapter (Абстрактный интерфейс)
**Назначение**: Единый интерфейс для провайдеров LLM

**Интерфейс**:
```python
from abc import ABC, abstractmethod

class LLMProviderAdapter(ABC):
    @abstractmethod
    def generate_completion(
        self, 
        prompt: str, 
        image: Optional[bytes],
        language: str,
        max_tokens: int
    ) -> str:
        """Генерирует текстовый ответ"""
        pass
    
    @abstractmethod
    def generate_structured_output(
        self, 
        prompt: str, 
        schema: dict,
        language: str
    ) -> dict:
        """Генерирует структурированный ответ по схеме"""
        pass
```

**Реализации**:
- `LLaVAAdapter`: Использует LLaVA (Large Language and Vision Assistant) через Ollama
- `QwenVLAdapter`: Использует Qwen-VL как альтернативную модель
- Поддержка квантизации 4-bit для работы на GPU с ограниченной памятью (4GB VRAM)

## Модели данных

### DeviceAnalysisResult
```python
from pydantic import BaseModel
from typing import List, Optional

class DeviceAnalysisResult(BaseModel):
    device_type: str
    confidence: float  # 0.0 - 1.0
    brand: Optional[str]
    model: Optional[str]
    suggested_categories: List[str]
    detected_controls: List[Control]
```

### Control
```python
class Control(BaseModel):
    id: str
    type: str  # button, knob, switch, lever, etc.
    label: Optional[str]
    bounding_box: BoundingBox
    confidence: float
```

### BoundingBox
```python
class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
```

### Message
```python
from enum import Enum

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime
    image_ref: Optional[str]  # Ссылка на изображение, если есть
```

### Session
```python
class Session(BaseModel):
    id: str
    user_id: str
    device_type: str
    device_image_url: str
    messages: List[Message]
    created_at: datetime
    updated_at: datetime
    device_context: DeviceContext
```

### DeviceContext
```python
class DeviceContext(BaseModel):
    device_type: str
    brand: Optional[str]
    model: Optional[str]
    detected_controls: List[Control]
    safety_warnings: List[str]
```

### Explanation
```python
class Step(BaseModel):
    number: int
    description: str
    warning: Optional[str]
    highlighted_area: Optional[BoundingBox]
    completed: bool = False

class Explanation(BaseModel):
    text: str
    steps: Optional[List[Step]]
    warnings: List[str]
    confidence: float
    sources: List[str]  # Ссылки на источники информации
```

## Свойства корректности

*Свойство — это характеристика или поведение, которое должно выполняться во всех допустимых выполнениях системы — по сути, формальное утверждение о том, что должна делать система. Свойства служат мостом между спецификациями, понятными человеку, и гарантиями корректности, проверяемыми машиной.*

## Локальные модели и оптимизация

### Computer Vision модели

**YOLOv8n (Nano):**
- Размер: ~6MB
- Назначение: Детекция объектов и элементов управления
- Inference time: ~20-30ms на Quadro P1000
- VRAM: ~500MB

**CLIP ViT-B/32:**
- Размер: ~350MB
- Назначение: Классификация типов устройств (zero-shot)
- Inference time: ~50-100ms
- VRAM: ~1GB

**EasyOCR:**
- Размер: ~100MB (English + Russian + Chinese)
- Назначение: Распознавание текста на кнопках и панелях
- Inference time: ~200-500ms
- VRAM: ~500MB

### Vision-Language модели

**LLaVA-7B (4-bit quantized):**
- Размер: ~4GB
- Назначение: Генерация объяснений и инструкций
- Inference time: ~2-5 секунд для ответа
- VRAM: ~2.5GB
- Через Ollama для упрощенного управления

**Qwen-VL-Chat (4-bit quantized):**
- Размер: ~4GB
- Назначение: Альтернативная модель для мультиязычности
- Inference time: ~2-5 секунд
- VRAM: ~2.5GB

### Стратегия управления памятью

1. **Lazy Loading**: Модели загружаются по требованию
2. **Model Unloading**: Неиспользуемые модели выгружаются через 5 минут простоя
3. **Batch Processing**: Группировка запросов для эффективности
4. **Quantization**: 4-bit квантизация для LLM моделей
5. **Mixed Precision**: FP16 для CV моделей

