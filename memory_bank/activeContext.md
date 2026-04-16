# Текущий фокус

## Активные задачи

- Интеграция Frontend и Backend через FastAPI API-слой
- Проверка контрактов для LLM/CV запросов и ответов
- Zustand store для управления состоянием сессий

## Приоритеты

1. Проверка и валидация существующих компонентов
2. Интеграция Frontend и Backend
3. Настройка LLM интеграции (LLaVA через Ollama)

## Текущие решения

- Backend: FastAPI с PyTorch CUDA на NVIDIA Quadro P1000
- Frontend: React + TypeScript + Vite + Tailwind CSS
- ML: YOLOv8n (легковесная модель для 4GB VRAM)
- LLM: LLaVA-7B-q4 через Ollama
- API-слой backend вынесен в `backend/app/api/routes.py` и подключен через router с префиксом `/api`

## Неопределенности

- Требуется проверка работы CUDA на текущей конфигурации
- Интеграция Frontend с Backend API требует реализации клиента на frontend
- Настройка автоматической выгрузки неиспользуемых моделей
