from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Backend
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    environment: str = "development"

    # CORS
    cors_origins: str = "http://localhost:3000"

    # Providers
    cv_provider: str = "google_vision"
    llm_provider: str = "openai"

    # Google Cloud Vision
    google_cloud_project_id: str = ""
    google_application_credentials: str = ""

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4-vision-preview"

    # Google Gemini
    google_gemini_api_key: str = ""
    google_gemini_model: str = "gemini-pro-vision"

    # Application
    max_image_size_mb: int = 10
    session_timeout_minutes: int = 30
    cache_ttl_hours: int = 24

    class Config:
        env_file = ".env"


settings = Settings()
