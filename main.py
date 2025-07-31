import cv2
import mediapipe as mp
import math
import numpy as np
import os

# Инициализация MediaPipe Hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Флаги для управления видео
scream_triggered = False
scream_video = None
# Добавляем переменную для сохранения силы глитча
persistent_noise_strength = 0

cap = cv2.VideoCapture(0)  # веб-камера

def remove_green_screen(frame, lower_green=None, upper_green=None):
    """Удаляет зеленый фон и делает его прозрачным"""
    # Диапазоны зеленого цвета в HSV
    if lower_green is None:
        lower_green = np.array([40, 40, 40])  # Нижний порог зеленого
    if upper_green is None:
        upper_green = np.array([80, 255, 255])  # Верхний порог зеленого
    
    # Конвертируем в HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Создаем маску для зеленого цвета
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Инвертируем маску (зеленый = 0, остальное = 255)
    mask_inv = cv2.bitwise_not(mask)
    
    # Применяем морфологические операции для очистки маски
    kernel = np.ones((3,3), np.uint8)
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel)
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel)
    
    # Размываем края маски для плавности
    mask_inv = cv2.GaussianBlur(mask_inv, (3, 3), 0)
    
    return mask_inv

def blend_with_background(foreground, background, mask):
    """Смешивает передний план с фоном используя маску"""
    # Нормализуем маску
    mask_norm = mask.astype(float) / 255
    
    # Расширяем маску до 3 каналов
    mask_3d = np.stack([mask_norm, mask_norm, mask_norm], axis=2)
    
    # Смешиваем изображения
    result = foreground * mask_3d + background * (1 - mask_3d)
    
    return result.astype(np.uint8)

def apply_glitch_effects(frame, noise_strength):
    """Применяет глитч-эффекты к кадру"""
    if noise_strength > 30:
        # Плавная вероятность инвертирования
        invert_prob = min((noise_strength - 20) / 80, 1.0)
        if np.random.rand() < invert_prob * 0.7:
            frame = cv2.bitwise_not(frame)

        # Глитч-эффект
        glitch_frame = frame.copy()
        num_glitches = int(1 + (noise_strength - 20) / 20)
        max_shift = int(5 + (noise_strength - 20) * 1.5)
        for _ in range(num_glitches):
            y = np.random.randint(0, frame.shape[0])
            h_strip = np.random.randint(5, 20)
            shift = np.random.randint(-max_shift, max_shift)
            glitch_frame[y:y+h_strip] = np.roll(glitch_frame[y:y+h_strip], shift, axis=1)
        frame = glitch_frame
    
    return frame

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    background_frame = frame.copy()  # Сохраняем фон веб-камеры

    # Если сработал триггер на дистанции 200, показываем видео вместо веб-камеры
    if scream_triggered:
        if scream_video is None:
            # Инициализируем видео
            if os.path.exists("scream.mp4"):
                scream_video = cv2.VideoCapture("scream.mp4")
                
                # Включаем звук в видео (если он есть)
                # OpenCV не воспроизводит звук, но можно попробовать через ffmpeg
                try:
                    import subprocess
                    # Запускаем воспроизведение звука в фоновом режиме
                    subprocess.Popen([
                        'ffplay', '-nodisp', '-autoexit', '-volume', '100', 'scream.mp4'
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except:
                    # Если ffplay недоступен, пробуем через pygame
                    try:
                        import pygame
                        pygame.mixer.init()
                        # Ищем отдельный аудиофайл
                        audio_file = None
                        for ext in ['.wav', '.mp3', '.ogg']:
                            if os.path.exists(f"scream{ext}"):
                                audio_file = f"scream{ext}"
                                break
                        
                        if audio_file:
                            pygame.mixer.music.load(audio_file)
                            pygame.mixer.music.set_volume(1.0)
                            pygame.mixer.music.play()
                    except:
                        print("Не удалось воспроизвести звук")
            else:
                print("Файл scream.mp4 не найден!")
                scream_triggered = False
        
        # Читаем кадр из видео
        if scream_video is not None:
            ret, video_frame = scream_video.read()
            if ret:
                # Масштабируем видео под размер окна веб-камеры
                h, w = background_frame.shape[:2]
                video_frame = cv2.resize(video_frame, (w, h))
                
                # Удаляем зеленый фон
                mask = remove_green_screen(video_frame)
                
                # Смешиваем видео с фоном веб-камеры
                frame = blend_with_background(video_frame, background_frame, mask)
            else:
                # Видео закончилось, возвращаемся к веб-камере
                scream_video.release()
                scream_video = None
                scream_triggered = False
                # Сбрасываем силу глитча после окончания видео
                persistent_noise_strength = 0
    
    # Обработка руки только если не показываем видео
    if not scream_triggered:
        # Конвертация BGR в RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Детекция руки
        results = hands.process(frame_rgb)

        noise_strength = 0
        current_distance = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Получение координат кончиков пальцев
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Координаты в пикселях
                h, w, _ = frame.shape
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

                # Рисуем круги на кончиках пальцев
                cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 0), -1)
                cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)

                # Линия между пальцами
                cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 0), 2)

                # Вычисление расстояния
                distance = math.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)
                current_distance = int(distance)
                
                # Изменяем цвет текста при приближении к 200
                text_color = (0, 0, 255)  # красный по умолчанию
                if current_distance >= 180:
                    text_color = (0, 255, 255)  # желтый при приближении
                if current_distance >= 200:
                    text_color = (0, 0, 255)  # красный при достижении
                
                # cv2.putText(frame, f"Distance: {current_distance}", (10, 30), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

                # Проверяем, достигли ли мы дистанции 200
                if current_distance >= 200:
                    scream_triggered = True

                # Чем больше дистанция, тем сильнее шум
                noise_strength = min(int(distance // 2), 100)
                # Сохраняем силу глитча для использования во время видео
                persistent_noise_strength = noise_strength

                # Отрисовка всех landmarks руки
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Применяем глитч-эффекты независимо от состояния видео
    # Используем текущую силу шума или сохраненную, если видео воспроизводится
    current_noise = persistent_noise_strength if scream_triggered else (noise_strength if 'noise_strength' in locals() else 0)
    
    # Применяем глитч-эффекты
    frame = apply_glitch_effects(frame, current_noise)

    # Масштабирование
    scale_factor = 1.5
    h, w = frame.shape[:2]
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    cv2.imshow("Hand Tracking", frame_resized)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Очистка ресурсов
if scream_video is not None:
    scream_video.release()
cap.release()
cv2.destroyAllWindows()

