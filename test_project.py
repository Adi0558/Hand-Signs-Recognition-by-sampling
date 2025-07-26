import numpy as np
from output import load_models, predict_gesture, LEFT_LABEL_MAP, RIGHT_LABEL_MAP

def test_label_maps():
    assert 'a' in LEFT_LABEL_MAP
    assert '1' in RIGHT_LABEL_MAP
    assert LEFT_LABEL_MAP['a'].startswith("Left")
    assert RIGHT_LABEL_MAP['1'].startswith("Right")

def test_model_loading():
    model_left, model_right = load_models()
    assert model_left is not None
    assert model_right is not None

def test_prediction_output_structure():
    # Load model just to pass a real one (doesn’t need to be accurate here)
    model_left, _ = load_models()

    # Create dummy input (21 landmarks x and y = 42 features)
    dummy_input = np.random.rand(1, 42)

    label, color = predict_gesture(model_left, dummy_input, LEFT_LABEL_MAP, "Left")

    assert isinstance(label, str)
    assert isinstance(color, tuple)
    assert len(color) == 3  # RGB color tuple
