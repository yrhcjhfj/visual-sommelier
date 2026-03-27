# Стек технологий

## Окружение

- **OS**: Windows
- **Python**: 3.11+
- **Node.js**: 18+
- **GPU**: NVIDIA Quadro P1000 (4GB VRAM, Compute Capability 6.1)
- **CUDA**: 11.8+ с cuDNN

## Backend зависимости

- FastAPI (веб-фреймворк)
- PyTorch + torchvision (ML фреймворк)
- ultralytics (YOLOv8)
- easyocr (распознавание текста)
- httpx (клиент для Ollama API)
- pydantic (валидация данных)

## Frontend зависимости

- React 18
- TypeScript
- Vite (сборщик)
- Tailwind CSS (стилизация)
- Zustand (управление состоянием)
- IndexedDB (локальное хранилище)

## ML модели

- **YOLOv8n**: ~6MB (детекция объектов)
- **EasyOCR**: ~100MB (OCR для EN/RU/ZH)
- **LLaVA-7B-q4**: ~4GB (через Ollama, квантизация 4-bit)

## Ограничения

- 4GB VRAM — все модели должны быть легковесными
- FP16 для CV моделей
- 4-bit квантизация для LLM
- Автоматическая выгрузка неиспользуемых моделей через 5 минут

## Окружение разработки

- VS Code с расширениями
- .env файл для конфигурации
- .kiro/ для спецификаций проекта
