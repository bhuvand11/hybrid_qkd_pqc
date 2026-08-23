"""
Memory-safe loader + preprocessor for the Kyber Pair-Pointwise Multiplication
side-channel dataset (Reference-PPM).

Start with a SMALL traces_no (e.g. 500) to confirm everything works before
scaling up. Saves processed data as .npy files so you never need to re-parse
the .mat files again.
"""

import copy
import numpy as np
import scipy.io

# ---------------------------------------------------------------------------
# CONFIG - adjust these
# ---------------------------------------------------------------------------
FILENAME = "pqc/data/Reference-PPM"   # folder containing the .mat files (forward slashes work on Windows too)
TRACES_NO = 5000                      # scale up once the small run confirms everything works. Max = 100000.
TRACES_PER_FILE = 100
N_FEATS = 50000                       # points per trace (fixed by the dataset)

# Which secret-related columns to build your label from:
#   columns 0,1 -> secret key bytes   (direct "is the key leaking" framing)
#   columns 2,3 -> input value
#   columns 4,5 -> output value
#
# IMPORTANT: the trained kyber_cnn_model was built on the OUTPUT bytes (4,5),
# not the secret key bytes. Keep this at (4,5) so any accuracy check against
# that model uses matching labels. Change only if you're training a new model
# from scratch and deliberately want a different leakage target.
LABEL_COLS = (4, 5)

# ---------------------------------------------------------------------------


def load_raw(traces_no, traces_per_file, n_feats):
    file_no = int(traces_no / traces_per_file)
    set_size = copy.deepcopy(traces_per_file)

    # float32 instead of float64 -> half the memory, negligible accuracy impact
    trace_kyber = np.zeros((file_no * set_size, n_feats), dtype=np.float32)
    vals = np.zeros((file_no * set_size, 12), dtype=np.int32)

    for ii in range(file_no):
        idx = np.arange(ii * set_size, (ii + 1) * set_size)

        path = f"{FILENAME}/tracesA{(ii + 1) * set_size - 1}.mat"
        mat = scipy.io.loadmat(path, squeeze_me=True)
        trace_kyber[idx, :] = np.array(mat["tracesA"], dtype=np.float32)

        path = f"{FILENAME}/noncesA{(ii + 1) * set_size - 1}.mat"
        mat = scipy.io.loadmat(path, squeeze_me=True)
        vals[idx, :] = np.array(mat["noncesA"], dtype=np.int32)

        print(f"Loaded file batch {ii + 1}/{file_no}")

    return trace_kyber, vals


def hamming_weight_labels(vals, cols):
    """Computes the Hamming Weight class (0-16) from two byte columns."""
    n = vals.shape[0]
    hw = np.zeros(n, dtype=np.int32)
    for j in range(n):
        hw[j] = bin(int(vals[j, cols[0]])).count("1") + bin(int(vals[j, cols[1]])).count("1")
    return hw


if __name__ == "__main__":
    print(f"Loading {TRACES_NO} traces (this may take a while)...")
    traces, vals = load_raw(TRACES_NO, TRACES_PER_FILE, N_FEATS)

    print(f"Computing Hamming Weight labels (cols {LABEL_COLS})...")
    labels = hamming_weight_labels(vals, LABEL_COLS)

    print(f"Traces shape: {traces.shape}  (traces x sample points)")
    print(f"Labels shape: {labels.shape}")
    print("Label distribution (class: count):")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  HW={u}: {c}")

    # --- Save processed arrays so you never need to touch the .mat files again ---
    np.save("kyber_traces.npy", traces)
    np.save("kyber_labels.npy", labels)
    np.save("kyber_vals.npy", vals)
    print("\nSaved kyber_traces.npy, kyber_labels.npy, kyber_vals.npy")
    print("Load them next time with: np.load('kyber_traces.npy')")