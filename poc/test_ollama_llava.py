"""
POC Test 0.1: Ollama + LLaVA
Проверка работы LLaVA для генерации описаний устройств
"""

import time
import base64
from pathlib import Path
import ollama
from PIL import Image


def encode_image(image_path: str) -> str:
    """Кодирует изображение в base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def test_llava_basic():
    """Базовый тест LLaVA"""
    print("=" * 60)
    print("POC Test 0.1: Ollama + LLaVA")
    print("=" * 60)
    
    # Проверка доступности Ollama
    try:
        models = ollama.list()
        print(f"✓ Ollama доступен. Установлено моделей: {len(models.get('models', []))}")
        
        # Проверка наличия LLaVA
        llava_found = any('llava' in m.get('name', '').lower() for m in models.get('models', []))
        if llava_found:
            print("✓ LLaVA модель найдена")
        else:
            print("✗ LLaVA модель не найдена. Запустите: ollama pull llava:7b-v1.6-mistral-q4_0")
            return False
            
    except Exception as e:
        print(f"✗ Ошибка подключения к Ollama: {e}")
        print("  Убедитесь что Ollama запущен: ollama serve")
        return False
    
    # Создаем тестовое изображение если его нет
    test_image_path = Path("test_images/test_device.jpg")
    if not test_image_path.exists():
        print("\n⚠ Тестовое изображение не найдено")
        print("  Создайте папку test_images/ и добавьте фото бытового устройства")
        print("  Например: пульт, стиральная машина, микроволновка")
        return False
    
    print(f"\n📷 Используем изображение: {test_image_path}")
    
    # Тест 1: Простое описание изображения
    print("\n--- Тест 1: Описание изображения ---")
    prompt = "Опиши что ты видишь на этом изображении. Отвечай на русском языке."
    
    start_time = time.time()
    try:
        response = ollama.chat(
            model='llava:7b-v1.6-mistral-q4_0',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [str(test_image_path)]
            }]
        )
        elapsed = time.time() - start_time
        
        answer = response['message']['content']
        print(f"\n⏱ Время ответа: {elapsed:.2f} секунд")
        print(f"\n💬 Ответ LLaVA:\n{answer}")
        
        if elapsed > 10:
            print(f"\n⚠ Время ответа превышает 10 секунд ({elapsed:.2f}s)")
        else:
            print(f"\n✓ Время ответа приемлемое")
            
    except Exception as e:
        print(f"\n✗ Ошибка при генерации: {e}")
        return False
    
    # Тест 2: Вопрос о функциях устройства
    print("\n--- Тест 2: Вопрос о функциях ---")
    prompt = "Какие элементы управления ты видишь на этом устройстве? Опиши их назначение. Отвечай на русском языке."
    
    start_time = time.time()
    try:
        response = ollama.chat(
            model='llava:7b-v1.6-mistral-q4_0',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [str(test_image_path)]
            }]
        )
        elapsed = time.time() - start_time
        
        answer = response['message']['content']
        print(f"\n⏱ Время ответа: {elapsed:.2f} секунд")
        print(f"\n💬 Ответ LLaVA:\n{answer}")
        
    except Exception as e:
        print(f"\n✗ Ошибка при генерации: {e}")
        return False
    
    # Тест 3: Инструкция по использованию
    print("\n--- Тест 3: Генерация инструкции ---")
    prompt = "Как пользоваться этим устройством? Дай пошаговую инструкцию. Отвечай на русском языке."
    
    start_time = time.time()
    try:
        response = ollama.chat(
            model='llava:7b-v1.6-mistral-q4_0',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [str(test_image_path)]
            }]
        )
        elapsed = time.time() - start_time
        
        answer = response['message']['content']
        print(f"\n⏱ Время ответа: {elapsed:.2f} секунд")
        print(f"\n💬 Ответ LLaVA:\n{answer}")
        
    except Exception as e:
        print(f"\n✗ Ошибка при генерации: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ Тест 0.1 завершен успешно")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_llava_basic()
    exit(0 if success else 1)
