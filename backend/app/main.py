import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .utils.gpu_utils import gpu_manager
from .adapters.factory import AdapterFactory
from .config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown."""
    # Startup
    logger.info("Starting Visual Sommelier API")
    
    # Check GPU availability
    cuda_available = gpu_manager.check_cuda_availability()
    if cuda_available:
        logger.info("GPU acceleration enabled")
        if settings.log_gpu_stats:
            gpu_manager.log_gpu_stats()
    else:
        logger.warning("GPU not available, using CPU")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Visual Sommelier API")
    AdapterFactory.cleanup_all()
    gpu_manager.cleanup()


app = FastAPI(
    title="Visual Sommelier API",
    description="API для приложения Визуальный Сомелье",
    version="0.1.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Visual Sommelier API"}


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint with GPU status."""
    cuda_available = gpu_manager.check_cuda_availability()
    
    health_info = {
        "status": "healthy",
        "cuda_available": cuda_available
    }
    
    if cuda_available and settings.log_gpu_stats:
        gpu_stats = gpu_manager.get_gpu_stats()
        if gpu_stats:
            health_info["gpu"] = {
                "device": gpu_stats.device_name,
                "memory_used_mb": gpu_stats.used_memory_mb,
                "memory_total_mb": gpu_stats.total_memory_mb,
                "utilization_percent": gpu_stats.utilization_percent
            }
    
    return health_info
