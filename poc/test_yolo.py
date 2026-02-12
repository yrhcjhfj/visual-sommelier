"""
POC Test 0.2: YOLOv8n для детекции объектов
Проверка скорости и точности детекции элементов на устройствах
"""

import time
from pathlib import Path
import torch
from ultralytics import YOLO
import cv2
import numpy as np


def check_cuda():
    """Проверка доступности CUDA"""
    print("=" * 60)
    print("Проверка CUDA")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA доступна")
        print(f"  Устройство: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA версия: {torch.version.cuda}")
        print(f"  Всего памяти: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        return True
    else:
        print("✗ CUDA недоступна. Будет использован CPU (медленно)")
        return False


def test_yolo_detection():
    """Тест детекции YOLOv8n"""
    print("\n" + "=" * 60)
    print("POC Test 0.2: YOLOv8n Object Detection")
    print("=" * 60)
    
    cuda_available = check_cuda()
    device = 'cuda:0' if cuda_available else 'cpu'
    
    # Загрузка модели
    print(f"\n📦 Загрузка YOLOv8n...")
    start_time = time.time()
    try:
        model = YOLO('yolov8n.pt')
        model.to(device)
        load_time = time.time() - start_time
        print(f"✓ Модель загружена за {load_time:.2f} секунд")
    except Exception as e:
        print(f"✗ Ошибка загрузки модели: {e}")
        return False
    
    # Проверка тестового изображения
    test_image_path = Path("test_images/test_device.jpg")
    if not test_image_path.exists():
        print("\n⚠ Тестовое изображение не найдено")
        print("  Создайте папку test_images/ и добавьте фото бытового устройства")
        return False
    
    print(f"\n📷 Используем изображение: {test_image_path}")
    
    # Загрузка изображения
    image = cv2.imread(str(test_image_path))
    if image is None:
        print(f"✗ Не удалось загрузить изображение")
        return False
    
    print(f"  Размер: {image.shape[1]}x{image.shape[0]}")
    
    # Warmup (первый запуск всегда медленнее)
    print("\n🔥 Warmup...")
    _ = model(image, verbose=False)
    
    # Тест производительности (5 запусков)
    print("\n⚡ Тест производительности (5 запусков)...")
    times = []
    for i in range(5):
        start_time = time.time()
        results = model(image, verbose=False)
        elapsed = time.time() - start_time
        times.append(elapsed * 1000)  # в миллисекундах
        print(f"  Запуск {i+1}: {elapsed*1000:.1f} ms")
    
    avg_time = np.mean(times)
    print(f"\n⏱ Среднее время: {avg_time:.1f} ms")
    
    if avg_time > 100:
        print(f"⚠ Время превышает 100ms")
    else:
        print(f"✓ Производительность приемлемая")
    
    # Анализ результатов детекции
    print("\n🔍 Результаты детекции:")
    results = model(image, verbose=False)[0]
    
    if len(results.boxes) == 0:
        print("  Объекты не обнаружены")
    else:
        print(f"  Обнаружено объектов: {len(results.boxes)}")
        
        # Группировка по классам
        classes = {}
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            confidence = float(box.conf[0])
            
            if cls_name not in classes:
                classes[cls_name] = []
            classes[cls_name].append(confidence)
        
        print("\n  Детали:")
        for cls_name, confidences in classes.items():
            avg_conf = np.mean(confidences)
            print(f"    {cls_name}: {len(confidences)} шт. (уверенность: {avg_conf:.2f})")
    
    # Сохранение результата с аннотациями
    output_path = Path("results/yolo_detection.jpg")
    output_path.parent.mkdir(exist_ok=True)
    
    annotated = results.plot()
    cv2.imwrite(str(output_path), annotated)
    print(f"\n💾 Результат сохранен: {output_path}")
    
    # Проверка использования GPU памяти
    if cuda_available:
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
        print(f"\n🎮 Использование GPU памяти:")
        print(f"  Выделено: {memory_allocated:.1f} MB")
        print(f"  Зарезервировано: {memory_reserved:.1f} MB")
    
    print("\n" + "=" * 60)
    print("✓ Тест 0.2 завершен успешно")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_yolo_detection()
    exit(0 if success else 1)
