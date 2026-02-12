"""
POC Test 0.4: Интеграционный end-to-end тест
Полный цикл: изображение → YOLO → OCR → LLaVA → ответ
"""

import time
from pathlib import Path
import torch
from ultralytics import YOLO
import easyocr
import ollama
import cv2
import numpy as np


def check_gpu_memory():
    """Проверка использования GPU памяти"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"🎮 GPU память:")
        print(f"  Выделено: {memory_allocated:.2f} GB")
        print(f"  Зарезервировано: {memory_reserved:.2f} GB")
        print(f"  Всего: {total_memory:.2f} GB")
        print(f"  Свободно: {total_memory - memory_reserved:.2f} GB")
        
        return memory_allocated, memory_reserved, total_memory
    return 0, 0, 0


def test_integration():
    """Интеграционный тест всего pipeline"""
    print("=" * 60)
    print("POC Test 0.4: End-to-End Integration Test")
    print("=" * 60)
    
    # Проверка CUDA
    if not torch.cuda.is_available():
        print("⚠ CUDA недоступна. Тест будет медленным на CPU")
    else:
        print(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
    
    # Проверка тестового изображения
    test_image_path = Path("test_images/test_device.jpg")
    if not test_image_path.exists():
        print("\n✗ Тестовое изображение не найдено")
        print("  Создайте test_images/test_device.jpg")
        return False
    
    print(f"\n📷 Изображение: {test_image_path}")
    
    # Начальное состояние GPU
    print("\n--- Начальное состояние ---")
    check_gpu_memory()
    
    total_start = time.time()
    
    # Шаг 1: Загрузка и детекция YOLOv8n
    print("\n" + "=" * 60)
    print("Шаг 1: YOLOv8n - Детекция объектов")
    print("=" * 60)
    
    step_start = time.time()
    try:
        print("📦 Загрузка YOLOv8n...")
        yolo_model = YOLO('yolov8n.pt')
        if torch.cuda.is_available():
            yolo_model.to('cuda:0')
        
        print("🔍 Детекция объектов...")
        image = cv2.imread(str(test_image_path))
        yolo_results = yolo_model(image, verbose=False)[0]
        
        step_time = time.time() - step_start
        print(f"⏱ Время: {step_time:.2f} сек")
        print(f"✓ Обнаружено объектов: {len(yolo_results.boxes)}")
        
        # Детали детекции
        if len(yolo_results.boxes) > 0:
            classes = {}
            for box in yolo_results.boxes:
                cls_name = yolo_model.names[int(box.cls[0])]
                if cls_name not in classes:
                    classes[cls_name] = 0
                classes[cls_name] += 1
            
            for cls_name, count in classes.items():
                print(f"  - {cls_name}: {count}")
        
        check_gpu_memory()
        
    except Exception as e:
        print(f"✗ Ошибка YOLO: {e}")
        return False
    
    # Шаг 2: Распознавание текста EasyOCR
    print("\n" + "=" * 60)
    print("Шаг 2: EasyOCR - Распознавание текста")
    print("=" * 60)
    
    step_start = time.time()
    try:
        print("📦 Инициализация EasyOCR...")
        reader = easyocr.Reader(
            ['en'],
            gpu=torch.cuda.is_available(),
            verbose=False
        )
        
        print("🔍 Распознавание текста...")
        # Load image and convert to RGB (EasyOCR expects RGB)
        img = cv2.imread(str(test_image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ocr_results = reader.readtext(img_rgb)
        
        step_time = time.time() - step_start
        print(f"⏱ Время: {step_time:.2f} сек")
        print(f"✓ Обнаружено текстовых блоков: {len(ocr_results)}")
        
        if len(ocr_results) > 0:
            print("\n📝 Распознанный текст:")
            for i, result in enumerate(ocr_results[:5], 1):  # Первые 5
                # result is a tuple: (bbox, text, confidence)
                text = result[1] if len(result) > 1 else str(result)
                conf = result[2] if len(result) > 2 else 0.0
                print(f"  {i}. '{text}' ({conf:.2f})")
            if len(ocr_results) > 5:
                print(f"  ... и еще {len(ocr_results) - 5}")
        
        check_gpu_memory()
        
    except Exception as e:
        import traceback
        print(f"✗ Ошибка OCR: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False
    
    # Шаг 3: Генерация объяснения LLaVA
    print("\n" + "=" * 60)
    print("Шаг 3: LLaVA - Генерация объяснения")
    print("=" * 60)
    
    step_start = time.time()
    try:
        # Формируем контекст из результатов YOLO и OCR
        context_parts = []
        
        if len(yolo_results.boxes) > 0:
            detected_objects = set()
            for box in yolo_results.boxes:
                detected_objects.add(yolo_model.names[int(box.cls[0])])
            context_parts.append(f"Обнаружены объекты: {', '.join(detected_objects)}")
        
        if len(ocr_results) > 0:
            texts = [result[1] for result in ocr_results]
            context_parts.append(f"Распознанный текст: {', '.join(texts[:5])}")
        
        context = ". ".join(context_parts) if context_parts else ""
        
        prompt = f"""Ты - помощник для понимания бытовых устройств. 
        
На изображении показано устройство. {context}

Ответь на русском языке:
1. Что это за устройство?
2. Какие основные элементы управления ты видишь?
3. Как пользоваться этим устройством? (кратко, 2-3 шага)

Будь конкретным и полезным."""
        
        print("💬 Отправка запроса в LLaVA...")
        response = ollama.chat(
            model='llava:7b-v1.6-mistral-q4_0',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [str(test_image_path)]
            }]
        )
        
        step_time = time.time() - step_start
        answer = response['message']['content']
        
        print(f"⏱ Время: {step_time:.2f} сек")
        print(f"\n✓ Ответ получен ({len(answer)} символов)")
        print("\n" + "=" * 60)
        print("📋 ОТВЕТ LLAVA:")
        print("=" * 60)
        print(answer)
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Ошибка LLaVA: {e}")
        print("  Убедитесь что Ollama запущен: ollama serve")
        return False
    
    # Итоговая статистика
    total_time = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 60)
    print(f"⏱ Общее время: {total_time:.2f} секунд")
    print(f"\n📊 Разбивка:")
    print(f"  - YOLOv8n: ~{step_time:.2f}s")
    print(f"  - EasyOCR: ~{step_time:.2f}s")  
    print(f"  - LLaVA: ~{step_time:.2f}s")
    
    print(f"\n🎯 Критерии успеха:")
    if total_time < 15:
        print(f"  ✓ Общее время < 15 сек ({total_time:.2f}s)")
    else:
        print(f"  ⚠ Общее время > 15 сек ({total_time:.2f}s)")
    
    if torch.cuda.is_available():
        _, memory_reserved, total_memory = check_gpu_memory()
        if memory_reserved < 3.5:
            print(f"  ✓ VRAM < 3.5 GB ({memory_reserved:.2f} GB)")
        else:
            print(f"  ⚠ VRAM > 3.5 GB ({memory_reserved:.2f} GB)")
    
    print("\n" + "=" * 60)
    print("✓ Интеграционный тест завершен успешно")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_integration()
    exit(0 if success else 1)
