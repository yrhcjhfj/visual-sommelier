# Текущий фокус

## Активные задачи

- Настройка LLM интеграции (LLaVA через Ollama)
- Проверка end-to-end работы с реальными устройствами

## Завершено

- Интеграция Frontend и Backend через FastAPI API-слой ✓
- Zustand store для управления состоянием сессий ✓
- Vite proxy для /api → http://localhost:8000 ✓

## Приоритеты

1. LLM интеграция (LLaVA через Ollama) - VS-04
2. End-to-end тестирование
3. Документация API

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
