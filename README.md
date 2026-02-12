# Визуальный Сомелье

Приложение для помощи пользователям в понимании сложных бытовых устройств с использованием компьютерного зрения и LLM.

## Структура проекта

```
visual-sommelier/
├── frontend/          # React + TypeScript приложение
├── backend/           # Python + FastAPI сервер
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
- Pydantic
- Google Cloud Vision API / OpenCV
- OpenAI GPT-4 Vision / Google Gemini

## Установка и запуск

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Переменные окружения

Скопируйте `.env.example` в `.env` и заполните необходимые ключи API.
