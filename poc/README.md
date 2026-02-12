# Proof of Concept - Тестирование работоспособности

Этот каталог содержит скрипты для проверки работоспособности основной идеи проекта перед полной разработкой.

## Цель

Проверить что локальные модели на NVIDIA Quadro P1000 могут:
1. Распознавать бытовые устройства
2. Детектировать элементы управления
3. Читать текст на устройствах
4. Генерировать понятные объяснения на русском языке
5. Работать с приемлемой скоростью

## Структура

- `test_ollama_llava.py` - Тест LLaVA через Ollama
- `test_yolo.py` - Тест YOLOv8n для детекции
- `test_easyocr.py` - Тест EasyOCR для распознавания текста
- `test_integration.py` - End-to-end интеграционный тест
- `test_images/` - Тестовые изображения устройств
- `results/` - Результаты тестирования

## Установка

### 1. Ollama

```bash
# Скачайте и установите с https://ollama.ai/download
# Запустите Ollama
ollama serve

# В другом терминале загрузите LLaVA
ollama pull llava:7b-v1.6-mistral-q4_0
```

### 2. Python зависимости

```bash
cd poc
pip install -r requirements.txt
```

## Запуск тестов

```bash
# Тест LLaVA
python test_ollama_llava.py

# Тест YOLO
python test_yolo.py

# Тест EasyOCR
python test_easyocr.py

# Полный интеграционный тест
python test_integration.py
```

## Критерии успеха

- LLaVA генерирует осмысленные ответы на русском < 10 сек
- YOLOv8n детектирует объекты < 100ms
- EasyOCR распознает текст на кириллице
- Общее использование VRAM < 3.5GB
- End-to-end время < 15 секунд
