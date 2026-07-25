"""
BB84 QKD simulation for generating a labeled QBER dataset.

This script simulates BB84 key exchange rounds under two conditions:
  - "normal" rounds: only realistic channel noise present
  - "attack" rounds: an eavesdropper (Eve) performs an intercept-resend attack

Each round produces one QBER value. Running many rounds gives you a labeled
dataset you can feed into an ML model (Isolation Forest, LSTM, etc.)

Install requirements first:
    pip install qiskit qiskit-aer pandas numpy --break-system-packages
"""

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

simulator = AerSimulator()


def bb84_round(n_qubits, noise_prob=0.02, eve_present=False, eve_intercept_prob=1.0):
    """
    Simulates one BB84 round between Alice and Bob, with Eve optionally present.
    Returns the QBER (fraction of mismatched bits in the sifted key) for this round.
    """
    rng = np.random.default_rng()

    alice_bits = rng.integers(0, 2, n_qubits)
    alice_bases = rng.integers(0, 2, n_qubits)   # 0 = Z (rectilinear), 1 = X (diagonal)
    bob_bases = rng.integers(0, 2, n_qubits)

    bob_results = np.zeros(n_qubits, dtype=int)

    for i in range(n_qubits):
        qc = QuantumCircuit(1, 1)

        # --- Alice encodes her bit in her chosen basis ---
        if alice_bits[i] == 1:
            qc.x(0)
        if alice_bases[i] == 1:
            qc.h(0)

        # --- Eve's intercept-resend attack (if active) ---
        if eve_present and rng.random() < eve_intercept_prob:
            eve_basis = rng.integers(0, 2)
            if eve_basis == 1:
                qc.h(0)
            qc.measure(0, 0)          # Eve measures, collapsing the qubit
            qc.reset(0)                # she resends a fresh qubit
            if eve_basis == 1:
                qc.h(0)                # re-encoded in the basis SHE guessed (may be wrong)

        # --- Realistic channel noise (random bit-flip) ---
        if rng.random() < noise_prob:
            qc.x(0)

        # --- Bob measures in his own randomly chosen basis ---
        if bob_bases[i] == 1:
            qc.h(0)
        qc.measure(0, 0)

        result = simulator.run(qc, shots=1, memory=True).result()
        bob_results[i] = int(result.get_memory()[0])

    # --- Sifting: keep only positions where Alice and Bob used the same basis ---
    matching = alice_bases == bob_bases
    sifted_alice = alice_bits[matching]
    sifted_bob = bob_results[matching]

    if len(sifted_alice) == 0:
        return None  # extremely unlikely, but guard against an empty sifted key

    errors = int(np.sum(sifted_alice != sifted_bob))
    qber = errors / len(sifted_alice)
    return qber


def generate_dataset(n_rounds=1000, qubits_per_round=100, attack_fraction=0.3):
    """
    Generates a labeled dataset of QBER values.
    label = 0 -> normal round, label = 1 -> Eve was present (attack round)
    """
    records = []
    for t in range(n_rounds):
        is_attack = np.random.rand() < attack_fraction
        qber = bb84_round(
            n_qubits=qubits_per_round,
            noise_prob=0.02,
            eve_present=is_attack,
        )
        if qber is not None:
            records.append({"time_step": t, "qber": qber, "label": int(is_attack)})

        if t % 100 == 0:
            print(f"Generated {t}/{n_rounds} rounds...")

    return pd.DataFrame(records)


if __name__ == "__main__":
    # NOTE: qubits_per_round * n_rounds = total simulated qubits.
    # Start small (as below) to confirm everything works, then scale up.
    # 500 rounds x 50 qubits = 25,000 single-qubit circuits -> a few minutes on a laptop/Colab.
    df = generate_dataset(n_rounds=500, qubits_per_round=50, attack_fraction=0.3)

    df.to_csv("../outputs/qber_timeseries_dataset.csv", index=False)

    print("\nDone.")
    print(df.head())
    print(f"\nTotal usable rounds: {len(df)}")
    print(f"Mean QBER (normal rounds): {df[df.label == 0].qber.mean():.4f}")
    print(f"Mean QBER (attack rounds): {df[df.label == 1].qber.mean():.4f}")