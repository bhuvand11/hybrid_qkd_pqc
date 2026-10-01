# Hybrid QKD–PQC Trust Assessment Framework

**Physical-layer AI/ML detection of eavesdropping and side-channel leakage for quantum-safe key security**

A capstone research project (B.Tech CSE, NMIMS Mukesh Patel School of Technology Management & Engineering, Mumbai) that builds two independent machine-learning detectors — one for Quantum Key Distribution (QKD) eavesdropping, one for Post-Quantum Cryptography (PQC) side-channel leakage — and fuses them into a single, real-time Accept/Flag/Reject key-trust decision.

---

## Table of Contents

- [Motivation](#motivation)
- [What this project actually does](#what-this-project-actually-does)
- [Architecture](#architecture)
- [Repository structure](#repository-structure)
- [QKD branch](#qkd-branch)
- [PQC branch](#pqc-branch)
- [Fusion layer](#fusion-layer)
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
3. **Fuses both branches** into a single Accept / Flag / Reject decision per key-establishment session, combining a live per-session QKD risk score with a one-time, device-level PQC leakage-risk estimate via log-odds combination and a cost-sensitive reject-option rule (see [Fusion layer](#fusion-layer)).

Both branches are fully synthetic-data / public-dataset based — the entire pipeline runs on a laptop or a free-tier Colab GPU, with no quantum hardware or specialized lab equipment required.

---

## Architecture

![System architecture](docs/figures/architecture_diagram.png)

Two branches run independently end-to-end and converge into a fusion decision:

- **QKD branch (blue)** — decoy-state BB84 simulator → Clean/IR/PNS attack injection → feature engineering (QBER, decoy anomaly, Y₁/e₁ bounds, secure key rate) → within-run normalization → sequence windowing → LSTM classifier (benchmarked against static-threshold, logistic-regression, and random-forest baselines).
- **PQC branch (maroon)** — Kyber power-trace corpus → per-trace normalization + downsampling → stratified split → 1D CNN classifier (benchmarked against a PCA + Random Forest baseline) → 17-class Hamming-weight leakage prediction.
- **Hybrid Trust Assessment Layer (green)** — combines both branches' signals into a final Accept / Flag / Reject key-trust decision. *This is the project's core novel contribution — see [Fusion layer](#fusion-layer).*

A vector version of the diagram (for papers/slides) is at [`docs/figures/architecture_diagram.pdf`](docs/figures/architecture_diagram.pdf); the generator script is in the project's scratch tooling and can be regenerated with matplotlib if the diagram needs edits.

---

## Repository structure

```
hybrid_qkd_pqc/
├── qkd/                                     # QKD eavesdropping-detection branch
│   ├── make_seeds.py                        # generates the 7 random seeds / run scenarios
│   ├── generate_qkd_dataset.py              # decoy-state BB84 simulator → per-run telemetry
│   ├── run_leakage_sweep.py                 # λ (PNS-to-QBER leakage) sensitivity sweep, static baselines
│   ├── run_leakage_sweep_lstm.py            # same sweep, retrained LSTM at each λ
│   ├── test.py                              # pulses-per-tick ablation runner
│   ├── generation_diagnostics_log.txt       # literature sanity-check log for the simulator
│   ├── regen_full_dataset.log               # full-scale dataset regeneration log
│   ├── qkd_lstm_training_STEP4.ipynb        # LSTM training + baseline comparison + evaluation
│   ├── qkd_lstm_training_STEP4_0309.ipynb   # earlier dated snapshot of the above
│   ├── single_seed_results.json             # per-feature AUCs, feature importances, leakage sanity check
│   └── multiseed_summary_results.json       # 7-seed robustness summary (mean ± std per method)
│
├── pqc/                                     # PQC side-channel-leakage branch
│   ├── scripts/
│   │   ├── PQC.ipynb                        # full CNN pipeline: load → preprocess → split → train → evaluate → export
│   │   ├── PQC_0309.ipynb                   # earlier dated snapshot of the above
│   │   ├── load_and_preprocess_kyber_dataset.py
│   │   ├── preprocess.py
│   │   ├── infer_kyber_cnn.py               # run inference with the trained model
│   │   └── batch_eval_kyber_cnn.py          # batch evaluation utility
│   └── models/kyber_cnn_export/
│       ├── kyber_cnn_model.keras            # trained CNN model
│       ├── model_metadata.json              # architecture, split sizes, achieved accuracy
│       ├── preprocessing_config.json        # exact preprocessing parameters used
│       ├── label_mapping.json
│       ├── baseline_pca_rf_results.json     # PCA+Random-Forest classical baseline results
│       ├── training_history.png
│       ├── training_history_stacked.png
│       └── confusion_matrix.png
│
├── fusion/                                  # Hybrid trust-fusion layer
│   ├── fusion_baseline.py                   # v1: bootstrap-pairing combiner (rejected design — see below)
│   ├── fusion_baseline_results.json
│   ├── sanity_checks.py                     # diagnostics that caught v1's label-circularity bug
│   ├── sanity_check_results.json
│   ├── fusion_v2_device_risk.py             # v2: corrected architecture (live QKD risk + static PQC device risk)
│   ├── fusion_v2_results.json               # session-level fused trust scores (current results)
│   ├── decision_policy.py                   # Chow's reject-option rule → Accept/Flag/Reject + cost sensitivity sweep
│   └── decision_policy_results.json
│
├── requirements.txt                         # Python dependencies (see Setup)
└── README.md                                # you are here
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

These are **within-run normalized** (median/MAD scaling) before modeling — pooling raw values across seeds/runs was found to introduce a confound that inflated apparent detectability.

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

## Fusion layer

Both branches above produce a risk signal, but they are not directly comparable: the QKD branch outputs a **live, per-session** probability (recomputed every session from ~2,879 sliding-window LSTM predictions), while the PQC branch outputs a **one-time, device-level** certification (how leaky this specific hardware is, estimated once from a held-out trace set and unchanged session to session unless the hardware is re-profiled). Treating them as the same kind of quantity was the project's first fusion mistake, and getting that right is what `fusion/` actually does.

**v1 — `fusion_baseline.py` (rejected design).** The first attempt treated both branches as freshly-drawn per-session random variables and combined them by bootstrap-pairing independent samples of QKD test-session outcomes with PQC test-trace outcomes. This produced a confusing, internally circular evaluation: per-trace PQC correctness can only ever be checked against the true secret value, which is available offline during lab profiling but never during live deployment, so it cannot validly stand in for a fresh per-session signal.

**Sanity check — `sanity_checks.py`.** Re-running the pipeline's diagnostics caught a second bug in the same v1 design: the "compromised" label used to evaluate the fused score and the confidence score used to *predict* it were the same quantity, which structurally guarantees high AUC regardless of whether the fusion logic is any good (an inflated, spurious ≈0.985). Both issues are documented in the script headers rather than silently fixed, since the trail is as informative as the final numbers.

**v2 — `fusion_v2_device_risk.py` (current design).** QKD stays exactly what it already was — a real-time, per-session LSTM detector, Platt-calibrated and aggregated over a session's windows by simple mean into $\rho_{QKD}$ (session-level AUC-ROC: **1.0000** on the 50-session test set, zero overlap between clean and attacked sessions). PQC becomes a fixed **device-level risk floor** with a proper Wilson 95% confidence interval, since it is estimated from a finite test set: $\rho_{PQC} = 0.9564$ `[0.9472, 0.9641]` for the measured leaky reference implementation, vs. $\rho_{PQC} = 0.4233$ `[0.4057, 0.4411]` for a lower-risk implementation used as a sensitivity stand-in. The two are combined via log-odds addition, which shifts every session's *absolute* trust by the same amount without changing their *relative* ranking — intentional, since fusion's job is to calibrate trust against a hardware risk the QKD branch cannot see, not to re-decide which session looks more suspicious.

Concretely, the cleanest session in the test set (QKD risk alone: 0.83%, essentially "safe") drops to **84.4% trust** once the measured device leakage is folded in, vs. **99.4%** under the lower-risk scenario with the identical QKD signal — the same session, judged very differently once hardware risk is accounted for.

**Decision policy — `decision_policy.py`.** The fused risk is converted into an Accept/Flag/Reject decision via Chow's (1970) classical reject-option rule, deriving two thresholds from an explicit, deployer-set cost model ($C_{FN}$: cost of accepting a compromised session, $C_{FP}$: cost of rejecting a clean one, $C_{REV}$: cheaper cost of manual review) rather than asserting one arbitrary cutoff. Across all 50 test sessions at an illustrative cost setting, the leaky reference device never receives Accept (0 clean, 0 attacked), splitting 14 clean/32 attacked into Flag and 18 clean/18 attacked into Reject; the lower-risk device instead Accepts all 18 clean sessions outright. This floor behavior — the leaky device can never fully clear, no matter how clean its QKD telemetry looks — holds across a wide sweep of cost ratios, not just the illustrative setting.

---

## Results so far

- **QKD branch:** LSTM sequence classifier evaluated against static-threshold, logistic-regression, and random-forest baselines (AUC-ROC / PR-AUC / F1). Best single-seed: 0.968 AUC-ROC vs. PNS (vs. 0.718 for a static threshold), matching the static threshold vs. IR (0.980 vs. 0.989). Confirmed robust across 7 independently-generated seeds (`qkd/multiseed_summary_results.json`), with dataset validity checked via literature-consistent sanity checks and an explicit confound diagnostic (seed/run-type independence from distance and key length).
- **PQC branch:** CNN reaches 95.64% test accuracy on a genuine held-out split, vs. 42.33% for a PCA+Random-Forest classical baseline and a 5.88% random-guess floor (`pqc/models/kyber_cnn_export/baseline_pca_rf_results.json`).
- **Fusion layer:** 1.0000 session-level AUC-ROC; fusing in device-level PQC risk shifts the cleanest session's trust from an implied "safe" 84.4%–99.4% depending on hardware leakage, and the Accept/Flag/Reject policy never fully clears a session on known-leaky hardware (`fusion/fusion_v2_results.json`, `fusion/decision_policy_results.json`).

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

**Fusion layer** (requires the exported QKD `.npy` logits/labels and the trained PQC CNN export to already exist):
```bash
cd fusion
python fusion_v2_device_risk.py     # corrected v1->v2 fusion: session-level QKD risk + device-level PQC risk
python decision_policy.py           # Chow's reject-option rule -> Accept/Flag/Reject + cost sensitivity sweep
# fusion_baseline.py + sanity_checks.py reproduce the rejected v1 design and the bug that sank it, for reference
```

---

## Novelty & positioning vs. existing work

**Genuinely novel:**
- An LSTM sequence model over QBER-and-decoy-statistic time series to catch attacks (including QBER-silent PNS) that a static threshold or single-timestep classifier cannot.
- A CNN producing a continuous, calibrated leakage-confidence signal for Kyber, rather than a binary pass/fail.
- The **fusion layer** combining a live per-session QKD risk score with a one-time device-level PQC risk estimate — via log-odds combination and a cost-sensitive reject-option rule — into one Accept/Flag/Reject trust decision. No surveyed paper fuses physical-layer signals from both sides this way (see [Literature survey](#literature-survey)); two of the most complete prior hybrid systems explicitly flag it as open future work.

**Explicitly not claimed as novel:**
- The CNN side-channel result itself (a known-achievable attack on a known-unprotected reference implementation, standard architecture).
- The decoy-state BB84 physics (Ma–Qi–Zhao–Lo bounds, GLLP secure key rate) — established theory, applied here to synthesize realistic training data.

**The gap in one line:** 20 surveyed hybrid QKD+PQC papers cluster around protocol design, key combiners, and network-level integration — none use both channels' own physical-layer diagnostic telemetry together as an ML-driven trust signal before the key is used.

---

## Known limitations & open items

- The QKD side is entirely simulated rather than measured from real hardware — the simulator is corrected against decoy-state/PNS-attack literature, but remains a model of reality. IR and PNS are also kept "pure" (one per run), so combined or out-of-distribution attack strategies are untested.
- The PQC branch uses 15,000 of the 100,000 available Kyber power traces, and no cross-device generalization test (analogous to the QKD branch's cooled/uncooled hardware split) has been run.
- The PCA+Random-Forest PQC baseline is trained on an 80/20 split that is not the same split object as the CNN's 70/15/15 split — comparable as a standalone baseline, but not an exact same-test-set comparison.
- The fusion layer's log-odds combination assumes the QKD channel and PQC implementation fail independently — physically reasonable, but unproven against a correlated failure mode (e.g. a shared supply-chain compromise).
- The Accept/Flag/Reject cost parameters ($C_{FN}$, $C_{FP}$, $C_{REV}$) are illustrative policy choices, not values elicited from a real deployment's risk tolerance.

---

## Roadmap

1. Test combined and out-of-distribution QKD attack strategies, and validate the simulator against real hardware telemetry.
2. Add a cross-device generalization test and a matched-split PCA+RF baseline for the PQC branch; scale to the full 100,000-trace dataset.
3. Validate the fusion layer's independence assumption against real joint QKD+PQC telemetry once such data becomes available.
4. Move from the current proof-of-concept fusion policy toward deployment-tuned cost parameters.

---

## Literature survey

A 20-paper survey (2020–2026) spanning IEEE Access, IEEE Photonics Journal, Advanced Quantum Technologies, EPJ Quantum Technology, and IEEE conferences (ICTON, ICAIC, QCNC, ICTBIG, DSD, ICECA, and others), grouped into four clusters:

- **(A) Hybrid QKD+PQC protocol / key-combiner design** — e.g. Garms et al. 2024 (Kyber+Falcon+QKD+PUF), Ricci et al. 2024 (3-key FPGA combiner), Aquina et al. 2024 (split-key PRF-XOR combiner)
- **(B) QKD eavesdropping detection / QBER-based security** — e.g. Waris et al. 2025 (HQNet-QKD, 96%+ classical detection)
- **(C) AI/ML applied to QKD or PQC individually** — e.g. Gupta & Mittal 2026 (AI/RL for QKD anomaly detection)
- **(D) Broader hybrid-architecture surveys** — e.g. Hettiarachchi et al. 2026 (80-study review, 5 architectural patterns)

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
