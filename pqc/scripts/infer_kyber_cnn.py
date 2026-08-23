"""
Part B — inference only. Loads the trained CNN and preprocessing config,
runs prediction on a power trace, and reports a leakage confidence score.
Does NOT retrain anything.
"""
import json
from pathlib import Path

import numpy as np
from tensorflow import keras

from preprocess import load_preprocessing_config, preprocess_traces

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "kyber_cnn_export"


def load_artifacts():
    model = keras.models.load_model(MODEL_DIR / "kyber_cnn_model.keras")
    config = load_preprocessing_config(MODEL_DIR / "preprocessing_config.json")
    with open(MODEL_DIR / "label_mapping.json") as f:
        label_mapping = json.load(f)
    return model, config, label_mapping


def predict_trace(raw_trace, model, config, label_mapping):
    """raw_trace: 1D numpy array, length = config['original_length'] (50000)."""
    processed = preprocess_traces(raw_trace, config)      # -> (1, 5000)
    cnn_input = processed[..., np.newaxis]                # -> (1, 5000, 1)

    probs = model.predict(cnn_input, verbose=0)[0]
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class] * 100)

    return {
        "predicted_hw": label_mapping[str(pred_class)],
        "confidence": confidence,
        "leakage_risk": "HIGH" if confidence > 50 else "LOW",
    }


if __name__ == "__main__":
    model, config, label_mapping = load_artifacts()
    print("Model + preprocessing config loaded successfully.")

    # Demo run on a real trace from your existing local dataset (not fabricated)
    traces = np.load(Path(__file__).resolve().parent / "kyber_traces.npy")
    sample = traces[0]

    result = predict_trace(sample, model, config, label_mapping)
    print(f"Predicted Hamming Weight: {result['predicted_hw']}")
    print(f"Confidence: {result['confidence']:.1f}%")