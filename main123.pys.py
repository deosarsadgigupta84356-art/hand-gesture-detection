import cv2
import mediapipe as mp
import pyautogui
import numpy as np

def fingers_up(hand_landmarks):
    finger_tips = [4, 8, 12, 16, 20]
    finger_pip = [3, 6, 10, 14, 18]

    fingers = []

    # Thumb
    if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_pip[0]].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other four fingers
    for tip, pip in zip(finger_tips[1:], finger_pip[1:]):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    gesture_text = "NO HAND"

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = fingers_up(hand_landmarks)

            h, w, _ = img.shape
            index_finger = hand_landmarks.landmark[8]
            x = int(index_finger.x * w)
            y = int(index_finger.y * h)

            thumb = hand_landmarks.landmark[4]
            thumb_y = int(thumb.y * h)
            distance = abs(thumb_y - y)

            gesture_text = "MOVE MODE"
            if distance < 25:
                gesture_text = "CLICK"

            screen_w, screen_h = pyautogui.size()
            screen_x = np.interp(x, [0, w], [0, screen_w])
            screen_y = np.interp(y, [0, h], [0, screen_h])

            if fingers[1] == 1:
                pyautogui.moveTo(screen_x, screen_y, duration=0.01)

            if fingers[1] == 1 and distance < 25:
                pyautogui.click()

    cv2.putText(
        img,
        gesture_text,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Hand Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()