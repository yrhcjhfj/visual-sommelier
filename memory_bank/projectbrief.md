# Визуальный Сомелье - Project Brief

## Цели проекта

Приложение для помощи пользователям в понимании сложных бытовых устройств с использованием компьютерного зрения и LLM. Работает полностью локально на NVIDIA Quadro P1000 с CUDA.

## Рамки проекта

- **Frontend**: React + TypeScript приложение с Tailwind CSS и Zustand
- **Backend**: Python + FastAPI сервер с PyTorch и CUDA
- **ML**: YOLOv8 (детекция), EasyOCR (распознавание текста), LLaVA (vision-language модель)
- **Локальная работа**: Все модели загружаются локально, без облачных API

## Project Deliverables

| ID | Deliverable | Status | Weight |
|----|-------------|--------|--------|
| VS-01 | Backend API сервер (FastAPI + конфигурация) | completed | 15 |
| VS-02 | CV Pipeline (YOLOv8 детекция объектов) | completed | 15 |
| VS-03 | OCR модуль (EasyOCR для EN/RU/ZH) | completed | 10 |
| VS-04 | LLM интеграция (LLaVA через Ollama) | in_progress | 15 |
| VS-05 | Frontend UI (React + Tailwind + Zustand) | in_progress | 20 |
| VS-06 | Интеграция всех компонентов | pending | 10 |
| VS-07 | POC тестирование и валидация | completed | 5 |
| VS-08 | Документация и README | completed | 10 |

**Сумма Weight**: 100 ✓
