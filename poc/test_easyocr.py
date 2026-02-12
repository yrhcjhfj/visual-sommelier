"""
POC Test 0.3: EasyOCR для распознавания текста
Проверка качества распознавания текста на устройствах
"""

import time
from pathlib import Path
import easyocr
import cv2
import numpy as np
import torch


def check_cuda():
    """Проверка доступности CUDA"""
    if torch.cuda.is_available():
        print(f"✓ CUDA доступна: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("✗ CUDA недоступна. Будет использован CPU")
        return False


def test_easyocr():
    """Тест распознавания текста EasyOCR"""
    print("=" * 60)
    print("POC Test 0.3: EasyOCR Text Recognition")
    print("=" * 60)
    
    cuda_available = check_cuda()
    
    # Инициализация EasyOCR
    print(f"\n📦 Инициализация EasyOCR (en, ru, zh)...")
    print("  ⚠ Первый запуск загрузит модели (~100MB)")
    
    start_time = time.time()
    try:
        reader = easyocr.Reader(
            ['en', 'ru', 'zh_sim'],
            gpu=cuda_available,
            verbose=False
        )
        load_time = time.time() - start_time
        print(f"✓ EasyOCR инициализирован за {load_time:.2f} секунд")
    except Exception as e:
        print(f"✗ Ошибка инициализации: {e}")
        return False
    
    # Проверка тестового изображения
    test_image_path = Path("test_images/test_device.jpg")
    if not test_image_path.exists():
        print("\n⚠ Тестовое изображение не найдено")
        print("  Создайте папку test_images/ и добавьте фото устройства с текстом")
        print("  Лучше всего: пульт с кнопками, панель управления")
        return False
    
    print(f"\n📷 Используем изображение: {test_image_path}")
    
    # Загрузка изображения
    image = cv2.imread(str(test_image_path))
    if image is None:
        print(f"✗ Не удалось загрузить изображение")
        return False
    
    print(f"  Размер: {image.shape[1]}x{image.shape[0]}")
    
    # Распознавание текста
    print("\n🔍 Распознавание текста...")
    start_time = time.time()
    try:
        results = reader.readtext(str(test_image_path))
        elapsed = time.time() - start_time
        print(f"⏱ Время распознавания: {elapsed:.2f} секунд")
    except Exception as e:
        print(f"✗ Ошибка распознавания: {e}")
        return False
    
    # Анализ результатов
    if len(results) == 0:
        print("\n⚠ Текст не обнаружен")
        print("  Возможные причины:")
        print("  - На изображении нет текста")
        print("  - Текст слишком мелкий или размытый")
        print("  - Плохое освещение")
    else:
        print(f"\n✓ Обнаружено текстовых блоков: {len(results)}")
        print("\n📝 Распознанный текст:")
        print("-" * 60)
        
        for i, (bbox, text, confidence) in enumerate(results, 1):
            print(f"{i}. '{text}' (уверенность: {confidence:.2f})")
        
        print("-" * 60)
        
        # Статистика по уверенности
        confidences = [conf for _, _, conf in results]
        avg_confidence = np.mean(confidences)
        min_confidence = np.min(confidences)
        
        print(f"\n📊 Статистика:")
        print(f"  Средняя уверенность: {avg_confidence:.2f}")
        print(f"  Минимальная уверенность: {min_confidence:.2f}")
        
        if avg_confidence < 0.5:
            print(f"  ⚠ Низкая уверенность распознавания")
        else:
            print(f"  ✓ Хорошая уверенность распознавания")
    
    # Визуализация результатов
    output_path = Path("results/ocr_result.jpg")
    output_path.parent.mkdir(exist_ok=True)
    
    # Рисуем bounding boxes и текст
    image_annotated = image.copy()
    for bbox, text, confidence in results:
        # Преобразуем bbox в целые числа
        pts = np.array(bbox, dtype=np.int32)
        
        # Рисуем прямоугольник
        cv2.polylines(image_annotated, [pts], True, (0, 255, 0), 2)
        
        # Добавляем текст
        x, y = pts[0]
        cv2.putText(
            image_annotated,
            f"{text} ({confidence:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    
    cv2.imwrite(str(output_path), image_annotated)
    print(f"\n💾 Результат сохранен: {output_path}")
    
    # Проверка использования GPU памяти
    if cuda_available:
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
        print(f"\n🎮 Использование GPU памяти:")
        print(f"  Выделено: {memory_allocated:.1f} MB")
        print(f"  Зарезервировано: {memory_reserved:.1f} MB")
    
    print("\n" + "=" * 60)
    print("✓ Тест 0.3 завершен успешно")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_easyocr()
    exit(0 if success else 1)
