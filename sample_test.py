import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

os.makedirs("models", exist_ok=True)

# ----------------------------- LEFT HAND MODEL -----------------------------
print("\n🖐️ Training LEFT hand model...")
left_data = pd.read_csv("data/left_hand.csv")
X_left = left_data.drop("label", axis=1)
y_left = left_data["label"]

Xl_train, Xl_test, yl_train, yl_test = train_test_split(X_left, y_left, test_size=0.2, random_state=42)


model_left = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    random_state=42
)
model_left.fit(Xl_train, yl_train)
left_accuracy = model_left.score(Xl_test, yl_test)
joblib.dump(model_left, "models/model_left.pkl")


joblib.dump(model_left, "models/model_left.pkl")
print(f"✅ Left-hand model trained. Accuracy: {left_accuracy:.2f}, Saved as: models/model_left.pkl")


# ----------------------------- RIGHT HAND MODEL -----------------------------
print("\n🖐️ Training RIGHT hand model...")
right_data = pd.read_csv("data/right_hand.csv")
X_right = right_data.drop("label", axis=1)
y_right = right_data["label"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_right, y_right, test_size=0.2, random_state=42)

model_right = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    random_state=42
)

model_right.fit(Xr_train, yr_train)
right_accuracy = model_right.score(Xr_test, yr_test)

joblib.dump(model_right, "models/model_right.pkl")
print(f"✅ Right-hand model trained. Accuracy: {right_accuracy:.2f}, Saved as: models/model_right.pkl")
