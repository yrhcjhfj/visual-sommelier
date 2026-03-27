# Текущий фокус

## Активные задачи

- Настройка Memory Bank проекта
- Инициализация системы отслеживания deliverables

## Приоритеты

1. Проверка и валидация существующих компонентов
2. Интеграция Frontend и Backend
3. Настройка LLM интеграции (LLaVA через Ollama)

## Текущие решения

- Backend: FastAPI с PyTorch CUDA на NVIDIA Quadro P1000
- Frontend: React + TypeScript + Vite + Tailwind CSS
- ML: YOLOv8n (легковесная модель для 4GB VRAM)
- LLM: LLaVA-7B-q4 через Ollama

## Неопределенности

- Требуется проверка работы CUDA на текущей конфигурации
- Интеграция Frontend с Backend API
- Настройка автоматической выгрузки неиспользуемых моделей
