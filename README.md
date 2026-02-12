# Визуальный Сомелье

Приложение для помощи пользователям в понимании сложных бытовых устройств с использованием компьютерного зрения и LLM. Работает полностью локально на NVIDIA Quadro P1000 с CUDA.

## Структура проекта

```
visual-sommelier/
├── frontend/          # React + TypeScript приложение
├── backend/           # Python + FastAPI сервер
├── models/            # Локальные ML модели
│   ├── cv/           # YOLOv8, CLIP, EasyOCR
│   └── llm/          # LLaVA, Qwen-VL
└── .kiro/            # Спецификации проекта
```

## Технологии

### Frontend
- React + TypeScript
- Vite
- Tailwind CSS
- Zustand (управление состоянием)
- IndexedDB (локальное хранилище)

### Backend
- Python 3.11+
- FastAPI
- PyTorch + CUDA
- YOLOv8 (детекция объектов)
- CLIP (классификация устройств)
- EasyOCR (распознавание текста)
- LLaVA через Ollama (vision-language модель)

### Системные требования

**GPU:**
- NVIDIA Quadro P1000 (4GB VRAM)
- CUDA 11.8+ и cuDNN
- Compute Capability 6.1+

**CPU/RAM:**
- 8GB RAM минимум (16GB рекомендуется)
- 4+ ядра CPU

**Хранилище:**
- 20GB для моделей
- SSD рекомендуется

## Установка

### 1. Установка CUDA и драйверов

```bash
# Проверьте версию CUDA
nvidia-smi

# Установите CUDA Toolkit 11.8+ если необходимо
# https://developer.nvidia.com/cuda-downloads
```

### 2. Установка Ollama (для LLaVA)

```bash
# Windows
# Скачайте с https://ollama.ai/download

# Запустите Ollama
ollama serve

# Загрузите LLaVA модель
ollama pull llava:7b-v1.6-mistral-q4_0
```

### 3. Backend

```bash
cd backend

# Создайте виртуальное окружение
python -m venv venv
venv\Scripts\activate  # Windows

# Установите зависимости
pip install -r requirements.txt

# Скопируйте .env.example в .env
copy ..\.env.example .env

# Запустите сервер
uvicorn app.main:app --reload
```

### 4. Frontend

```bash
cd frontend

# Установите зависимости
npm install

# Запустите dev сервер
npm run dev
```

## Модели

Модели загружаются автоматически при первом запуске:

- **YOLOv8n**: ~6MB (детекция объектов)
- **CLIP ViT-B/32**: ~350MB (классификация)
- **EasyOCR**: ~100MB (OCR для EN/RU/ZH)
- **LLaVA-7B-q4**: ~4GB (через Ollama)

## Использование GPU

Приложение автоматически использует GPU для всех операций:
- CV модели работают в FP16 для оптимизации
- LLM модели квантизованы до 4-bit
- Неиспользуемые модели выгружаются через 5 минут

Мониторинг GPU:
```bash
# Проверка использования
nvidia-smi

# Логи в приложении показывают VRAM usage
```

## Переменные окружения

Скопируйте `.env.example` в `.env` и настройте под свою систему.
