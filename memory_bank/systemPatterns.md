# Технические решения

## Архитектура

### Backend (Python + FastAPI)
- **Структура**: app/ с разделением на adapters/, api/, models/, services/, utils/
- **API**: RESTful endpoints через FastAPI routers
- **Маршруты**: `POST /api/analyze`, `POST /api/explain`, `POST /api/instructions`, `POST /api/clarify`
- **Модели**: PyTorch модели для CV и LLM
- **Конфигурация**: pyproject.toml + requirements.txt

### Frontend (React + TypeScript)
- **Структура**: src/ с components/, pages/, services/, store/, types/
- **Состояние**: Zustand store
- **Стилизация**: Tailwind CSS + PostCSS
- **Сборка**: Vite

### ML Pipeline
- **Детекция**: YOLOv8n (ultralytics) для обнаружения объектов
- **OCR**: EasyOCR для извлечения текста (EN/RU/ZH)
- **Анализ**: LLaVA через Ollama API для понимания контекста

## Паттерны

- **Адаптеры**: Модульная система адаптеров для работы с разными моделями
- **Сервисы**: Бизнес-логика вынесена в services/
- **API orchestration**: HTTP-слой валидирует входные данные, ограничения размера изображения и делегирует работу сервисам
- **Store**: Централизованное управление состоянием через Zustand

## Связи подсистем

```
Frontend (React) → Backend API (FastAPI) → CV Pipeline (YOLOv8 + EasyOCR)
                                         → LLM Service (LLaVA via Ollama)
```
