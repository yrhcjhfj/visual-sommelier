from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Backend
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    environment: str = "development"

    # CORS
    cors_origins: str = "http://localhost:3000"

    # GPU Configuration
    cuda_visible_devices: str = "0"
    torch_cuda_arch_list: str = "6.1"

    # Providers
    cv_provider: str = "yolo"
    cv_model_path: str = "./models/cv"
    llm_provider: str = "llava"
    llm_model_path: str = "./models/llm"

    # YOLOv8
    yolo_model: str = "yolov8n.pt"
    yolo_confidence: float = 0.5

    # CLIP
    clip_model: str = "ViT-B/32"

    # EasyOCR
    easyocr_languages: str = "en,ru,zh"

    # LLaVA (via Ollama)
    ollama_host: str = "http://localhost:11434"
    llava_model: str = "llava:7b-v1.6-mistral-q4_0"
    llava_max_tokens: int = 512
    llava_temperature: float = 0.7

    # Qwen-VL
    qwen_model: str = "Qwen-VL-Chat-Int4"
    qwen_max_tokens: int = 512

    # Model Management
    model_unload_timeout: int = 300
    enable_model_caching: bool = True
    use_mixed_precision: bool = True

    # Application
    max_image_size_mb: int = 10
    session_timeout_minutes: int = 30
    cache_ttl_hours: int = 24

    # Logging
    log_level: str = "INFO"
    log_gpu_stats: bool = True

    class Config:
        env_file = ".env"


settings = Settings()
