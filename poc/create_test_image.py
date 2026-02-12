import cv2
import numpy as np

# Создаем изображение пульта ДУ (простая имитация)
img = np.ones((600, 400, 3), dtype=np.uint8) * 50  # Темный фон

# Рисуем корпус пульта
cv2.rectangle(img, (100, 50), (300, 550), (80, 80, 80), -1)
cv2.rectangle(img, (100, 50), (300, 550), (120, 120, 120), 3)

# Рисуем кнопки с текстом
buttons = [
    (200, 100, "POWER"),
    (150, 180, "VOL+"),
    (250, 180, "CH+"),
    (150, 240, "VOL-"),
    (250, 240, "CH-"),
    (200, 320, "MENU"),
    (150, 400, "OK"),
    (200, 480, "EXIT"),
]

for x, y, text in buttons:
    cv2.circle(img, (x, y), 25, (200, 200, 200), -1)
    cv2.circle(img, (x, y), 25, (255, 255, 255), 2)
    
    # Добавляем текст
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 0.4, 1)[0]
    text_x = x - text_size[0] // 2
    text_y = y + text_size[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 0.4, (0, 0, 0), 1)

# Добавляем бренд
cv2.putText(img, "SAMSUNG", (130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

# Сохраняем
cv2.imwrite('test_images/test_device.jpg', img)
print("Test image created: test_images/test_device.jpg")
