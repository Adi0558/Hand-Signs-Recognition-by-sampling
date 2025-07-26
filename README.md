# Hand Sign Detection Using Sampling For Beginners #

#### Video Demo:  [URL HERE](https://youtu.be/1Vhx5azuoTY?si=j8o7TC3NnkcrgYKn)


#### Description:
This project is a real-time hand gesture recognition system designed to assist in interpreting the hand signs of mute individuals. Using Python, OpenCV, and MediaPipe, the system detects hand landmarks, classifies gestures based on pre-collected samples, and identifies whether the gesture is made by the left or right hand.

Each hand can be trained independently using sampled gesture data. The system then uses machine learning models (e.g., Random Forest) to classify gestures accurately. You can define custom gestures (like alphabets or common signs) for both hands and receive live predictions directly on the screen, overlaid with bounding boxes and gesture labels.

#### Key Features:
- 📷 Real-time hand tracking using MediaPipe
- ✋ Differentiates between Left and Right hand
- 🎯 Predicts gestures using trained ML models
- 🧠 Uses custom-trained classifiers (Random Forest by default)
- 📁 Saves training samples into CSV for both hands
- 🔒 Shows "Unknown" if the model is uncertain (below confidence threshold)
- 🔄 Easily extendable for A–Z gestures or numbers

#### How It Works:
1. **Sample Collection** – Capture hand landmarks and label them.
2. **Model Training** – Train separate models for left and right hands.
3. **Prediction** – Detect and classify gestures using camera input in real-time.

#### Libraries Used:
- Python 3.x
- OpenCV
- MediaPipe
- NumPy
- scikit-learn (RandomForestClassifier)
- joblib
- pytest (for basic testing)

