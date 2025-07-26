import cv2
import mediapipe as mp
import numpy as np
import joblib
from typing import Tuple, Dict

# Constants
CONFIDENCE_THRESHOLD = 0.80
LEFT_LABEL_MAP = {
    
    'b': 'Left: B',
    'c': 'Left: C',
    'h': 'Left: Hello',
    't': 'Left: Cool',
    's': 'Left: Stop',
    'c': 'Left: CS50P',
    '6': 'Left: 6',
    '7': 'Left: 7',
    '8': 'Left: 8',
    '9': 'Left: 9',
    '0': 'Left: 10'
}
RIGHT_LABEL_MAP = {
    'a': 'Right: A',
    'b': 'Right: B',
    'c': 'Right: C',
    't': 'Right: Take Care',
    's': 'Right: Stay there',
    'c': 'Right: Spider-man',
    'y': 'Right: Yes',
    'n': 'Right: No',
    '1': 'Right: 1',
    '2': 'Right: 2',
    '3': 'Right: 3',
    '4': 'Right: 4',
    '5': 'Right: 5'
}


def load_models() -> Tuple:
    """Load pre-trained gesture models for both hands."""
    model_left = joblib.load("models/model_left.pkl")
    model_right = joblib.load("models/model_right.pkl")
    return model_left, model_right


def get_landmark_input(hand_landmarks, normalize: bool = False) -> np.ndarray:
    """Extracts and optionally normalizes x/y coordinates from hand landmarks."""
    x = [lm.x for lm in hand_landmarks.landmark]
    y = [lm.y for lm in hand_landmarks.landmark]

    if normalize:
        x = [i - x[0] for i in x]
        y = [i - y[0] for i in y]

    return np.array(x + y).reshape(1, -1)


def predict_gesture(
    model, input_data: np.ndarray, label_map: Dict[str, str], hand_type: str
) -> Tuple[str, Tuple[int, int, int]]:
    """Predict gesture and return label and box color."""
    probabilities = model.predict_proba(input_data)[0]
    max_prob = np.max(probabilities)
    predicted_label = model.classes_[np.argmax(probabilities)]

    if max_prob >= CONFIDENCE_THRESHOLD:
        gesture_text = str(label_map.get(predicted_label, str(predicted_label)))  # <-- fixed
        box_color = (0, 255, 0)
    else:
        gesture_text = f"{hand_type}: Unknown"
        box_color = (0, 0, 255)

    return gesture_text, box_color



def main():
    model_left, model_right = load_models()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks and result.multi_handedness:
            for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                hand_type = result.multi_handedness[i].classification[0].label
                model = model_left if hand_type == "Left" else model_right
                label_map = LEFT_LABEL_MAP if hand_type == "Left" else RIGHT_LABEL_MAP

                # Prediction
                input_data = get_landmark_input(hand_landmarks)
                gesture_text, box_color = predict_gesture(model, input_data, label_map, hand_type)

                # Bounding box
                x_px = [int(lm.x * w) for lm in hand_landmarks.landmark]
                y_px = [int(lm.y * h) for lm in hand_landmarks.landmark]
                x_min, x_max = min(x_px), max(x_px)
                y_min, y_max = min(y_px), max(y_px)

                # Draw box and label
                cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), box_color, 2)
                cv2.putText(frame, gesture_text, (x_min - 20, y_min - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

                gesture_text = str(gesture_text)  # ensures it's always a string
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Two-Hand Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
