"""
Batch evaluation — runs the trained CNN on all traces in kyber_traces.npy
and reports accuracy against the ground-truth labels in kyber_labels.npy.
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


def predict_batch(raw_traces, model, config, label_mapping):
    """raw_traces: (n_traces, original_length) numpy array."""
    processed = preprocess_traces(raw_traces, config)       # -> (n, 5000)
    cnn_input = processed[..., np.newaxis]                  # -> (n, 5000, 1)

    probs = model.predict(cnn_input, verbose=1, batch_size=64)
    pred_classes = np.argmax(probs, axis=1)
    confidences = probs[np.arange(len(probs)), pred_classes] * 100

    pred_hw = np.array([label_mapping[str(c)] for c in pred_classes])
    return pred_hw, confidences


if __name__ == "__main__":
    model, config, label_mapping = load_artifacts()
    print("Model + preprocessing config loaded successfully.")

    traces = np.load(Path(__file__).resolve().parent / "kyber_traces.npy")
    true_labels = np.load(Path(__file__).resolve().parent / "kyber_labels.npy")

    print(f"Running inference on {traces.shape[0]} traces...")
    pred_hw, confidences = predict_batch(traces, model, config, label_mapping)

    # --- Accuracy metrics ---
    exact_match = pred_hw == true_labels
    within_1 = np.abs(pred_hw - true_labels) <= 1

    exact_acc = exact_match.mean() * 100
    within1_acc = within_1.mean() * 100
    mean_conf = confidences.mean()
    high_leak_frac = (confidences > 50).mean() * 100

    print("\n--- Results ---")
    print(f"Exact-match accuracy:      {exact_acc:.2f}%")
    print(f"Within ±1 HW accuracy:     {within1_acc:.2f}%")
    print(f"Mean prediction confidence:{mean_conf:.1f}%")
    print(f"Traces flagged HIGH leak:  {high_leak_frac:.1f}% (confidence > 50%)")

    # --- Per-class breakdown ---
    print("\nPer true-HW-class accuracy:")
    for hw_val in sorted(np.unique(true_labels)):
        mask = true_labels == hw_val
        n = mask.sum()
        acc = exact_match[mask].mean() * 100 if n > 0 else 0.0
        print(f"  HW={hw_val}: n={n:4d}  exact_acc={acc:5.1f}%")

    # --- Save full per-trace results for the fusion layer / report ---
    out_path = Path(__file__).resolve().parent / "kyber_cnn_predictions.npz"
    np.savez(out_path,
             predicted_hw=pred_hw,
             confidence=confidences,
             true_hw=true_labels)
    print(f"\nSaved per-trace predictions to {out_path.name}")