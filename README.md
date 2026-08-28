# Hybrid QKD–PQC Trust Assessment Framework

**Physical-layer AI/ML detection of eavesdropping and side-channel leakage for quantum-safe key security**

A capstone research project (B.Tech CSE, NMIMS Mukesh Patel School of Technology Management & Engineering, Mumbai) that builds two independent machine-learning detectors — one for Quantum Key Distribution (QKD) eavesdropping, one for Post-Quantum Cryptography (PQC) side-channel leakage — and works toward fusing them into a single, real-time key-trust decision.

---

## Table of Contents

- [Motivation](#motivation)
- [What this project actually does](#what-this-project-actually-does)
- [Architecture](#architecture)
- [Repository structure](#repository-structure)
- [QKD branch](#qkd-branch)
- [PQC branch](#pqc-branch)
- [Fusion layer (in progress)](#fusion-layer-in-progress)
- [Results so far](#results-so-far)
- [Setup & installation](#setup--installation)
- [How to reproduce the results](#how-to-reproduce-the-results)
- [Novelty & positioning vs. existing work](#novelty--positioning-vs-existing-work)
- [Known limitations & open items](#known-limitations--open-items)
- [Roadmap](#roadmap)
- [Literature survey](#literature-survey)
- [Tech stack](#tech-stack)
- [Authors](#authors)

---

## Motivation

Large-scale quantum computers will eventually break the RSA/ECC public-key cryptography that secures most of today's internet traffic. Two quantum-safe defenses are being deployed, often together, as a hedge against either one failing:

- **QKD** — a physics-based key exchange (e.g. BB84) where the act of eavesdropping itself perturbs the quantum channel, in principle making spying detectable.
- **PQC** — classical algorithms (e.g. Kyber, Dilithium — both now NIST standards) built on math problems believed hard even for quantum computers.

Hybrid **QKD + PQC** deployments (telecom pilots, testbeds like Berlin OpenQKD) combine both for defense-in-depth. But each half carries its own physical-layer weakness that is rarely monitored in practice:

| Layer | Known weakness | Current practice |
|---|---|---|
| QKD | Eavesdropping raises the Quantum Bit Error Rate (QBER) — but subtle attacks (photon-number-splitting) barely move it | A static QBER threshold (~11% for BB84): crude, and blind to attacks designed to stay under it |
| PQC | Kyber/Dilithium chips leak secret-dependent information through power/EM emissions during computation | Ad-hoc, early-stage countermeasures; even NIST-approved masked implementations have been broken in published attacks; no real-time way to quantify leakage |

**This project's question:** can a unified AI/ML framework, watching both channels' own physical-layer telemetry, provide a more adaptive, real-time trust signal than today's static, siloed checks?

---

## What this project actually does

1. **Simulates** a decoy-state BB84 QKD channel under three conditions — clean, intercept-resend (IR) attack, photon-number-splitting (PNS) attack — and trains an **LSTM** sequence model to tell them apart from time-series features, outperforming a static QBER threshold and classical ML baselines.
2. **Trains a 1D CNN** on a real Kyber power-trace dataset to classify Hamming-weight side-channel leakage, quantifying how much a reference Kyber implementation leaks through power consumption.
3. **Works toward a fusion layer** (in progress — see [Roadmap](#roadmap)) that will combine both branches' outputs into a single Accept / Flag / Reject decision for a given key-establishment session.

Both branches are fully synthetic-data / public-dataset based — the entire pipeline runs on a laptop or a free-tier Colab GPU, with no quantum hardware or specialized lab equipment required.

---

## Architecture

![System architecture](docs/figures/architecture_diagram.png)

Two branches run independently end-to-end and converge into a fusion decision:

- **QKD branch (blue)** — decoy-state BB84 simulator → Clean/IR/PNS attack injection → feature engineering (QBER, decoy anomaly, Y₁/e₁ bounds, secure key rate) → within-run normalization → sequence windowing → LSTM classifier (benchmarked against static-threshold, logistic-regression, and random-forest baselines).
- **PQC branch (maroon)** — Kyber power-trace corpus → per-trace normalization + downsampling → stratified split → 1D CNN classifier (benchmarked against a PCA + Random Forest baseline) → 17-class Hamming-weight leakage prediction.
- **Hybrid Trust Assessment Layer (green)** — combines both branches' signals into a final Accept / Flag / Reject key-trust decision. *This is the project's core novel contribution and its current milestone — see [Fusion layer](#fusion-layer-in-progress).*

A vector version of the diagram (for papers/slides) is at [`docs/figures/architecture_diagram.pdf`](docs/figures/architecture_diagram.pdf); the generator script is in the project's scratch tooling and can be regenerated with matplotlib if the diagram needs edits.

---

## Repository structure

```
hybrid_qkd_pqc/
├── qkd/                                  # QKD eavesdropping-detection branch
│   ├── make_seeds.py                     # generates the 7 random seeds / run scenarios
│   ├── generate_qkd_dataset.py           # decoy-state BB84 simulator → per-run CSVs
│   ├── run_leakage_sweep.py              # sensitivity sweep across attack strengths
│   ├── leakage_sweep_results.csv         # output of the sweep
│   ├── generation_diagnostics_log.txt    # literature sanity-check log for the simulator
│   ├── qkd_lstm_training_STEP4.ipynb     # LSTM training + baseline comparison + evaluation
│   ├── qkd_train_decoy_seed{1-7}.csv     # generated training runs, one file per seed
│   └── qkd_test_decoy_seed{1-7}.csv      # generated test runs, one file per seed
│
├── pqc/                                  # PQC side-channel-leakage branch
│   ├── data/
│   │   └── Reference-PPM/                # reference power-trace data
│   ├── scripts/
│   │   ├── PQC.ipynb                     # full CNN pipeline: load → preprocess → split → train → evaluate → export
│   │   ├── load_and_preprocess_kyber_dataset.py
│   │   ├── preprocess.py
│   │   ├── infer_kyber_cnn.py            # run inference with the trained model
│   │   ├── batch_eval_kyber_cnn.py       # batch evaluation utility
│   │   ├── kyber_traces.npy              # power traces
│   │   ├── kyber_labels.npy              # Hamming-weight labels
│   │   └── kyber_vals.npy                # associated intermediate values
│   ├── models/kyber_cnn_export/
│   │   ├── kyber_cnn_model.keras         # trained CNN model
│   │   ├── model_metadata.json           # architecture, split sizes, achieved accuracy
│   │   ├── preprocessing_config.json     # exact preprocessing parameters used
│   │   ├── label_mapping.json
│   │   ├── training_history.png
│   │   └── confusion_matrix.png
│   └── outputs/                          # evaluation outputs
│
├── docs/
│   └── figures/
│       ├── architecture_diagram.png      # system architecture diagram
│       └── architecture_diagram.pdf      # vector version (paper/slides)
│
├── CAPSTONE_CONTEXT.md                   # internal working notes / project state tracker
├── litreviewcapstone.md                  # 20-paper literature review table
├── eda_within_run_report.md              # exploratory data analysis notes (QKD confound diagnostics)
├── capstoneppt.md                        # earlier title-approval presentation content
├── requirements.txt                      # Python dependencies (see Setup)
└── README.md                             # you are here
```

---

## QKD branch

**Goal:** detect whether a QKD session is clean or under attack, using only the physical-layer telemetry a real QKD system already produces — without relying on a fixed threshold.

**Simulation.** A decoy-state BB84 channel is simulated with three weak-coherent-pulse intensities (signal μ = 0.5, decoy ν = 0.1, vacuum ω = 0), Poisson photon-number statistics, a fiber attenuation model, and a detector dark-count probability. Three run conditions are generated across 7 random seeds (50 train / 50 test runs each):

- **Clean** — no eavesdropper
- **Intercept-Resend (IR)** — Eve measures and resends every photon; raises QBER substantially (the "loud" attack)
- **Photon-Number-Splitting (PNS)** — Eve exploits multi-photon pulses; QBER stays *almost unchanged* (the "quiet" attack this project specifically targets)

**Features computed per run**, using standard decoy-state analysis (Ma–Qi–Zhao–Lo, 2005):

- QBER
- Y₁ lower bound / e₁ upper bound (single-photon yield and error-rate bounds)
- yield ratio, decoy anomaly score
- finite-key sifted-key-length statistics
- GLLP-style secure key rate

These are **within-run normalized** (median/MAD scaling) before modeling — pooling raw values across seeds/runs was found to introduce a confound that inflated apparent detectability (see `eda_within_run_report.md`).

**Model.** Features are windowed into 60-tick sequences (stride 30) per run and fed to a **2-layer LSTM** (hidden size 64, ~54.6K parameters, PyTorch, `BCEWithLogitsLoss` with `pos_weight` for class imbalance). Evaluated with AUC-ROC, PR-AUC, and F1 against three baselines: a static QBER threshold, logistic regression, and random forest.

**Validation.** Before any model was trained, the simulator's output was checked against literature-expected ranges (`generation_diagnostics_log.txt`):

| Metric | Observed | Expected |
|---|---|---|
| Clean QBER mean | 2.22% | 1–5% |
| Strong-IR QBER mean | 13.49% | > 11% (BB84 abort threshold) |
| PNS QBER mean | 2.34% | ≈ clean (PNS is designed to be QBER-silent) |
| Clean Y₁ lower | 0.10618 | — |
| PNS Y₁ lower | 0.10181 | ≪ clean |
| Clean yield ratio | 3.7561 | ≈ 5.0 absent dark counts |
| PNS yield ratio | 3.8672 | < clean |

This confirms the key finding: under PNS, QBER barely moves, but Y₁ and yield-ratio drop exactly as decoy-state theory predicts — the subtle signal a static threshold cannot see, and the LSTM is trained to catch.

---

## PQC branch

**Goal:** quantify how much a real Kyber implementation leaks through power consumption, as a continuous, trustworthy risk signal (not a binary broken/not-broken verdict).

**Dataset.** A public Kyber power-trace corpus — 150 trace files, 50,000 samples per trace, labeled with a 17-class Hamming-weight target.

**Preprocessing** (`load_and_preprocess_kyber_dataset.py`, `preprocess.py`):
- Per-trace z-score normalization
- Average-pooling downsample: 50,000 → 5,000 samples/trace
- Stratified 70/15/15 split (`random_state=42`): 10,500 train / 2,250 validation / 2,250 test

**Model** (`PQC.ipynb`). A 1D CNN — stacked Conv1D + BatchNorm + ReLU + MaxPool blocks → dense layers → 17-class softmax, ℓ₂-regularized, trained 80 epochs (TensorFlow/Keras 2.20). Benchmarked against a classical **PCA + Random Forest** baseline on the same downsampled traces.

**Result:**

| Metric | Value |
|---|---|
| Test accuracy (held-out stratified split) | **95.64%** |
| Best validation accuracy | 95.38% |
| Test loss | 0.711 |
| Classes | 17 (Hamming weight) |
| Train / Val / Test samples | 10,500 / 2,250 / 2,250 |

> **Note on this result:** this confirms the reference implementation is exploitably leaky via power traces — useful as a validated leakage-quantification input for the fusion layer. It replicates a known-achievable attack on a known-unprotected reference target using a standard CNN; it is *not* claimed as a novel side-channel-analysis technique. The project's novelty is the fusion architecture, not either branch's model in isolation.

An earlier evaluation script (`batch_eval_kyber_cnn.py`) reported a higher, **contaminated** figure (96.12%) because it evaluated on a sequential batch of traces that partially overlapped with the training set. The 95.64% figure above is the correct one, sourced from the notebook's own genuine stratified held-out split (`model_metadata.json`) — this discrepancy is documented here deliberately so it isn't silently reintroduced.

---

## Fusion layer (in progress)

The two branches above are complete and independently validated. **Not yet built:** the layer that combines a QKD-branch trust signal and a PQC-branch trust signal into one session-level Accept / Flag / Reject decision — this is the project's core novel contribution.

Open design questions:
- How to normalize/weight two outputs on different scales and with different semantics (a whole-run sequence-classifier confidence vs. a per-trace classification confidence)
- Whether to start with an interpretable rule-based combiner before attempting a learned fusion model
- How to define ground truth for "session trust," since no existing dataset labels joint QKD+PQC trustworthiness

---

## Results so far

- **QKD branch:** LSTM sequence classifier evaluated against static-threshold, logistic-regression, and random-forest baselines (AUC-ROC / PR-AUC / F1), with dataset validity confirmed via literature-consistent sanity checks and an explicit confound diagnostic (seed/run-type independence from distance and key length).
- **PQC branch:** CNN achieves 95.64% test accuracy on a genuine held-out split; PCA+RF classical baseline implemented (exact figure pending a rerun — see [Known limitations](#known-limitations--open-items)).
- **Fusion layer:** not yet evaluated — no results to report yet.

---

## Setup & installation

```bash
git clone <this-repo-url>
cd hybrid_qkd_pqc
pip install -r requirements.txt
```

`requirements.txt` covers the QKD simulation and general ML stack (NumPy, pandas, scikit-learn, Qiskit/Qiskit-Aer, TensorFlow). **PyTorch** (used only by the QKD LSTM notebook) is installed inline at the top of `qkd/qkd_lstm_training_STEP4.ipynb` — install it separately if running that notebook outside Colab:

```bash
pip install torch
```

---

## How to reproduce the results

**QKD branch:**
```bash
cd qkd
python make_seeds.py                # (re)generate the 7 seeds
python generate_qkd_dataset.py      # simulate decoy-state BB84 runs → CSVs
# then open qkd_lstm_training_STEP4.ipynb to train/evaluate the LSTM and baselines
```

**PQC branch:**
```bash
cd pqc/scripts
python infer_kyber_cnn.py           # run inference with the already-trained model
# or open PQC.ipynb to retrain the CNN from scratch and reproduce the 95.64% test result
```

---

## Novelty & positioning vs. existing work

**Genuinely novel:**
- An LSTM sequence model over QBER-and-decoy-statistic time series to catch attacks (including QBER-silent PNS) that a static threshold or single-timestep classifier cannot.
- A CNN producing a continuous leakage-confidence signal for Kyber, rather than a binary pass/fail.
- The **fusion architecture** combining both physical-layer signals into one session-level trust decision — no surveyed paper does this at the physical-layer signal level (see [Literature survey](#literature-survey)).

**Explicitly not claimed as novel:**
- The CNN side-channel result itself (a known-achievable attack on a known-unprotected reference implementation, standard architecture).
- The decoy-state BB84 physics (Ma–Qi–Zhao–Lo bounds, GLLP secure key rate) — established theory, applied here to synthesize realistic training data.

**The gap in one line:** 20 surveyed hybrid QKD+PQC papers cluster around protocol design, key combiners, and network-level integration — none use both channels' own physical-layer diagnostic telemetry together as an ML-driven trust signal before the key is used.

---

## Known limitations & open items

- PCA+Random Forest baseline accuracy for the PQC branch was computed in an earlier run but not saved — needs a rerun to log the exact number.
- QKD multi-seed robustness results (per-seed AUC/F1 averaged across all 7 seeds) are still being finalized.
- Of the 20 papers in the literature survey, 4 were read in full PDF with verified citation metadata (Waris et al. 2025, Garms et al. 2024, Gupta & Mittal 2026, Aquina et al. 2024); the remaining 16 need their full author lists / DOIs / page numbers verified before final submission.
- The fusion layer is unbuilt — see [Fusion layer](#fusion-layer-in-progress) and [Roadmap](#roadmap).

---

## Roadmap

1. Design and implement the fusion layer (core remaining deliverable).
2. Log the exact PCA+RF baseline accuracy for the PQC branch.
3. Complete and report multi-seed robustness results for the QKD branch.
4. Verify remaining literature citation details.
5. Finalize the research paper (Springer LNCS format) consolidating literature review, methodology, and results.

---

## Literature survey

A 20-paper survey (2020–2026) spanning IEEE Access, IEEE Photonics Journal, Advanced Quantum Technologies, EPJ Quantum Technology, and IEEE conferences (ICTON, ICAIC, QCNC, ICTBIG, DSD, ICECA, and others), grouped into four clusters:

- **(A) Hybrid QKD+PQC protocol / key-combiner design** — e.g. Garms et al. 2024 (Kyber+Falcon+QKD+PUF), Ricci et al. 2024 (3-key FPGA combiner), Aquina et al. 2024 (split-key PRF-XOR combiner)
- **(B) QKD eavesdropping detection / QBER-based security** — e.g. Waris et al. 2025 (HQNet-QKD, 96%+ classical detection)
- **(C) AI/ML applied to QKD or PQC individually** — e.g. Gupta & Mittal 2026 (AI/RL for QKD anomaly detection)
- **(D) Broader hybrid-architecture surveys** — e.g. Hettiarachchi et al. 2026 (80-study review, 5 architectural patterns)

Full table with objectives, contributions, challenges, and future scope for all 20 papers is in [`litreviewcapstone.md`](litreviewcapstone.md).

---

## Tech stack

| Layer | Tools |
|---|---|
| QKD simulation | Custom Python decoy-state BB84 simulator (deterministic, seed-controlled) |
| QKD modeling | PyTorch (LSTM), scikit-learn (logistic regression, random forest baselines) |
| PQC modeling | TensorFlow/Keras 2.20 (CNN), scikit-learn (PCA + random forest baseline) |
| Data | Self-generated QKD run CSVs; public Kyber Cortex-M4 power-trace dataset |
| Environment | Python 3, Jupyter/Colab notebooks, Git/GitHub |
| Diagrams/figures | matplotlib |

No specialized quantum or side-channel-acquisition hardware is required — the full pipeline is reproducible on a laptop or a free-tier Colab GPU.

---

## Authors

- Nirmit Desai
- Rishika Desai
- Bhuvan Devarakonda
- Vanshika Doshi

Guide: Dr. Pallavi Mangrulkar

NMIMS Mukesh Patel School of Technology Management & Engineering, Mumbai
