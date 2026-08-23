"""
Shared preprocessing — MUST exactly match what was used during CNN training
(z-score normalize per trace, then average-pool downsample).
"""
import json
import numpy as np


def load_preprocessing_config(config_path):
    with open(config_path) as f:
        return json.load(f)


def normalize_and_downsample(traces, factor):
    """traces: shape (n_traces, n_samples) or (n_samples,) for a single trace."""
    traces = np.atleast_2d(traces)
    mean = traces.mean(axis=1, keepdims=True)
    std = traces.std(axis=1, keepdims=True) + 1e-8
    normed = (traces - mean) / std

    n_traces, n_samples = normed.shape
    usable_len = (n_samples // factor) * factor
    trimmed = normed[:, :usable_len]
    return trimmed.reshape(n_traces, -1, factor).mean(axis=2)


def preprocess_traces(traces, config):
    return normalize_and_downsample(traces, config["downsample_factor"])