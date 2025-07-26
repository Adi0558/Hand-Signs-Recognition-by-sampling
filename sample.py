import cv2
import mediapipe as mp
import csv
import os

# Setup
os.makedirs("data", exist_ok=True)
left_file = open("data/left_hand.csv", 'w', newline='')
right_file = open("data/right_hand.csv", 'w', newline='')
left_writer = csv.writer(left_file)
right_writer = csv.writer(right_file)

# Header
header = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + ["label"]
left_writer.writerow(header)
right_writer.writerow(header)

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("Show hand gesture and press a key to label it (a-z, 0-9). Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks and result.multi_handedness:
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the frame first
    cv2.imshow("Collecting Samples", frame)

    # Read a single key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting sample collection.")
        break

    # If a valid gesture key is pressed, save data
    if key in [ord(c) for c in "abcdefghijklmnopqrstuvwxyz0123456789"]:
        gesture = chr(key)
        if result.multi_hand_landmarks and result.multi_handedness:
            for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                hand_type = result.multi_handedness[i].classification[0].label

                x_list = [lm.x for lm in hand_landmarks.landmark]
                y_list = [lm.y for lm in hand_landmarks.landmark]
                row = x_list + y_list + [gesture]

                if hand_type == "Left":
                    left_writer.writerow(row)
                    left_file.flush()
                    print("👈 Left hand sample saved:", gesture)
                elif hand_type == "Right":
                    right_writer.writerow(row)
                    right_file.flush()
                    print("👉 Right hand sample saved:", gesture)

# Cleanup
left_file.close()
right_file.close()
cap.release()
cv2.destroyAllWindows()
