"""
generate_qkd_dataset.py
========================
BB84 QKD eavesdropping-detection dataset generator with decoy-state protocol.
Incorporates three-intensity (signal/decoy/vacuum) WCP source physics and
Lo-Ma-Chen (2005) decoy-state analysis features for PNS-attack detection.

KEY ADDITIONS over the baseline script:
1. Three-intensity WCP source: signal (mu=0.5), decoy (mu=0.1), vacuum (mu=0.0)
2. Physics-based PNS attack: Eve blocks single-photon pulses with ONE shared
   probability across signal/decoy/vacuum intensities (she cannot tell them
   apart -- see the comment above the PNS block in simulate_long_timeline
   for the literature this is grounded in), plus a small secondary QBER
   leakage term representing realistic (non-idealized) attack imperfection.
3. Per-intensity gain/QBER observables
4. Lo-Ma-Chen (2005) decoy-state bounds
5. decoy_anomaly = expected_yield_ratio - actual_yield_ratio
6. Rolling windows on key decoy features (30-tick window).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

@dataclass
class HardwareProfile:
    name: str
    dark_count_rate_hz: float
    detector_efficiency: float
    dead_time_s: float
    detection_window_s: float = 3.5e-9

    def dark_count_prob_per_pulse(self) -> float:
        return 1 - np.exp(-self.dark_count_rate_hz * self.detection_window_s)


HARDWARE_PROFILES = {
    "cooled":   HardwareProfile("cooled",   dark_count_rate_hz=100,  detector_efficiency=0.85, dead_time_s=1e-6),
    "uncooled": HardwareProfile("uncooled", dark_count_rate_hz=6000, detector_efficiency=0.20, dead_time_s=50e-6),
}

FIBER_LOSS_DB_PER_KM = 0.2
INTRINSIC_MISALIGNMENT = 0.02

MU_SIGNAL = 0.5
MU_DECOY  = 0.1
MU_VACUUM = 0.0

FRAC_SIGNAL = 0.75
FRAC_DECOY  = 0.20
FRAC_VACUUM = 0.05

MEAN_PHOTON_NUMBER = MU_DECOY
F_EC = 1.16
EXPECTED_YIELD_RATIO = MU_SIGNAL / MU_DECOY

# How much of the blocked single-photon fraction leaks into QBER for a
# real (non-idealized) PNS attack. An idealized PNS attack is QBER-silent;
# Ashkenazy et al. 2024 measured nonzero QBER in a real PNS realization but
# did not report a generic coefficient we can cite directly -- this value
# is a plausible small-residual default, NOT a literature-derived constant.
# Treat it as a sensitivity parameter: sweep it (e.g. 0, 0.01, 0.02, 0.05)
# rather than trusting this single value.
PNS_QBER_LEAKAGE_FACTOR = 0.02

RUN_TYPE_PROBS = {"clean": 0.35, "pure_ir": 0.325, "pure_pns": 0.325}


def fiber_transmittance(distance_km):
    return 10 ** (-FIBER_LOSS_DB_PER_KM * distance_km / 10.0)


@dataclass
class AttackEpisode:
    kind: str
    strength: float
    start_tick: int
    end_tick: int
    ramp: bool


def sample_attack_episodes(
    rng,
    duration_s: int,
    forced_kind: str,
    min_episodes: int = 3,
    max_episodes: int = 8,
    episode_duration_range_s: tuple = (300, 3600),
    min_gap_s: int = 600,
) -> list:
    episodes = []
    cursor = 0
    n_ep = rng.integers(min_episodes, max_episodes + 1)
    for _ in range(n_ep):
        dur = int(rng.uniform(*episode_duration_range_s))
        gap = int(rng.uniform(min_gap_s, min_gap_s * 8))
        start = cursor + gap
        end = start + dur
        if end >= duration_s:
            break
        strength = (
            float(rng.beta(1.5, 4.0)) if forced_kind == "intercept_resend"
            else float(rng.uniform(0.01, 0.15))
        )
        ramp = bool(rng.integers(0, 2))
        episodes.append(AttackEpisode(forced_kind, strength, start, end, ramp))
        cursor = end
    return episodes


def episodes_to_arrays(episodes: list, duration_s: int):
    kind_arr = np.full(duration_s, "none", dtype=object)
    strength_arr = np.zeros(duration_s)
    for ep in episodes:
        span = ep.end_tick - ep.start_tick
        if not ep.ramp or span <= 2:
            strength_arr[ep.start_tick:ep.end_tick] = ep.strength
        else:
            t = np.arange(span) / span
            ramp_shape = np.clip(np.minimum(t / 0.33, (1 - t) / 0.33), 0, 1)
            strength_arr[ep.start_tick:ep.end_tick] = ep.strength * ramp_shape
        kind_arr[ep.start_tick:ep.end_tick] = ep.kind
    return kind_arr, strength_arr


def _binary_entropy(x: np.ndarray) -> np.ndarray:
    x_c = np.clip(x, 1e-10, 1.0 - 1e-10)
    return -x_c * np.log2(x_c) - (1.0 - x_c) * np.log2(1.0 - x_c)


def simulate_long_timeline(
    run_id: int,
    hardware_name: str,
    run_type: str,
    duration_s: int = 86_400,
    pulses_per_tick: int = 200_000,
    distance_km: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    pns_qber_leakage_factor: float = PNS_QBER_LEAKAGE_FACTOR, 
) -> pd.DataFrame:
    rng = rng or np.random.default_rng()
    hardware = HARDWARE_PROFILES[hardware_name]
    distance_km = distance_km if distance_km is not None else rng.uniform(5, 80)

    eta_fiber = fiber_transmittance(distance_km)
    eta_total  = eta_fiber * hardware.detector_efficiency
    dark_p     = hardware.dark_count_prob_per_pulse()

    t      = np.arange(duration_s)
    t_frac = t / duration_s

    p_vacuum_signal = np.exp(-MU_SIGNAL)
    p_single_signal = MU_SIGNAL * np.exp(-MU_SIGNAL)
    p_multi_signal  = 1.0 - p_vacuum_signal - p_single_signal

    p_vacuum_decoy  = np.exp(-MU_DECOY)
    p_single_decoy  = MU_DECOY * np.exp(-MU_DECOY)
    p_multi_decoy   = 1.0 - p_vacuum_decoy - p_single_decoy

    def _dead_time_factor(mu: float) -> float:
        raw_signal = mu * eta_total
        raw_total  = raw_signal + dark_p
        return 1.0 / (1.0 + raw_total * (hardware.dead_time_s / hardware.detection_window_s) * 1e-3)

    dtf_signal = _dead_time_factor(MU_SIGNAL)
    dtf_decoy  = _dead_time_factor(MU_DECOY)

    y1_signal_honest = p_single_signal * eta_total * dtf_signal
    ym_signal_honest = p_multi_signal  * eta_total * dtf_signal
    y1_decoy_honest  = p_single_decoy  * eta_total * dtf_decoy
    ym_decoy_honest  = p_multi_decoy   * eta_total * dtf_decoy

    Q_signal_honest = y1_signal_honest + ym_signal_honest + dark_p
    Q_decoy_honest  = y1_decoy_honest  + ym_decoy_honest  + dark_p
    Q_vacuum_honest = dark_p

    signal_click_p_compat = y1_decoy_honest + ym_decoy_honest

    n_harmonics = rng.integers(2, 4)
    drift = np.zeros(duration_s)
    for _ in range(n_harmonics):
        period_frac = rng.uniform(0.15, 1.0)
        amp  = rng.uniform(0.0, 0.004) / n_harmonics
        phase = rng.uniform(0, 2 * np.pi)
        drift += amp * np.sin(2 * np.pi * t_frac / period_frac + phase)
    misalignment_now = INTRINSIC_MISALIGNMENT + drift

    baseline_qber = (
        misalignment_now * signal_click_p_compat + 0.5 * dark_p
    ) / (signal_click_p_compat + dark_p + 1e-12)

    if run_type == "clean":
        episodes = []
    else:
        forced_kind = "intercept_resend" if run_type == "pure_ir" else "pns_like"
        episodes = sample_attack_episodes(rng, duration_s, forced_kind)

    kind_arr, strength_arr = episodes_to_arrays(episodes, duration_s)
    is_pns = (kind_arr == "pns_like").astype(float)
    is_ir  = (kind_arr == "intercept_resend").astype(float)

    # PNS attack model (v2, corrected against the decoy-state PNS literature):
    #
    # Eve performs a QND photon-number measurement, blocks single-photon
    # pulses, and losslessly forwards multi-photon pulses after extracting
    # a copy. Critically, Eve CANNOT distinguish signal from decoy pulses --
    # that indistinguishability is the core assumption decoy-state security
    # proofs rely on. As one review puts it: since Eve cannot tell signal
    # and decoy states apart, she performs the same attack on both, and it
    # is only the *differing intensities* that make the attack show up
    # differently in the resulting statistics (arXiv:0712.0517). So Eve
    # uses ONE shared per-photon block probability regardless of which
    # intensity produced the pulse -- she does not (cannot) recalibrate her
    # strategy per channel. The asymmetric effect on signal vs decoy gain
    # that decoy-state analysis is built to detect emerges entirely from
    # signal and decoy having different photon-number distributions to
    # begin with (different mu), not from Eve treating the channels
    # differently. (Previously this code scaled the block fraction
    # separately per channel by that channel's own p_multi/p_single ratio,
    # which implicitly assumed Eve could tell mu apart -- breaking the
    # premise decoy-state QKD's security proof depends on, and incidentally
    # suppressing the very signal-vs-decoy asymmetry decoy-state analysis
    # is supposed to expose.)
    pns_block_frac = np.clip(strength_arr, 0.0, 0.95)

    effective_Q_signal = (
        y1_signal_honest * (1.0 - pns_block_frac * is_pns)
        + ym_signal_honest
        + dark_p
    )
    effective_Q_decoy = (
        y1_decoy_honest * (1.0 - pns_block_frac * is_pns)
        + ym_decoy_honest
        + dark_p
    )
    effective_Q_vacuum = Q_vacuum_honest

    # An idealized PNS attack is QBER-silent by construction: Eve never
    # measures or disturbs the polarization/basis of photons she forwards,
    # so it introduces no errors detectable via QBER alone -- this is
    # exactly why decoy-state protocols were invented (QBER can't see it;
    # yield/gain statistics can). Real implementations do leak a small
    # amount of QBER as a side effect of the attack itself -- e.g. Ashkenazy
    # et al. 2024 (Adv. Quantum Technol.) measured nonzero QBER in an
    # experimental PNS realization, attributing it to unavoidable
    # perturbation of other properties (in their case, temporal) of the
    # forwarded mode. We model that as a small residual tied to how much of
    # the single-photon flux was actually blocked, well below the primary
    # loud-attack (intercept-resend) QBER signature, rather than as a
    # dominant, free-standing term.
    ir_contrib  = strength_arr / 4.0 * is_ir
    pns_contrib = pns_qber_leakage_factor * pns_block_frac * is_pns
    true_qber       = np.clip(baseline_qber + ir_contrib + pns_contrib, 0.0, 0.5)
    true_qber_decoy = np.clip(baseline_qber + ir_contrib + pns_contrib, 0.0, 0.5)

    n_signal_pulses_tick = pulses_per_tick * FRAC_SIGNAL
    n_decoy_pulses_tick  = pulses_per_tick * FRAC_DECOY
    n_vacuum_pulses_tick = pulses_per_tick * FRAC_VACUUM

    n_sifted_signal = rng.poisson(n_signal_pulses_tick * effective_Q_signal * 0.5)
    n_sifted_signal = np.clip(n_sifted_signal, 5, None)

    n_sifted_decoy = rng.poisson(n_decoy_pulses_tick * effective_Q_decoy * 0.5)
    n_sifted_decoy = np.clip(n_sifted_decoy, 2, None)

    n_sifted_vacuum = rng.poisson(n_vacuum_pulses_tick * effective_Q_vacuum * 0.5)

    n_errors_signal = rng.binomial(n_sifted_signal, true_qber)
    n_errors_decoy  = rng.binomial(n_sifted_decoy,  true_qber_decoy)
    qber_obs_signal = n_errors_signal / n_sifted_signal
    qber_obs_decoy  = n_errors_decoy  / n_sifted_decoy

    n_sifted      = n_sifted_signal
    observed_qber = qber_obs_signal

    yield_signal_obs = n_sifted_signal / (n_signal_pulses_tick * 0.5)
    yield_decoy_obs  = n_sifted_decoy  / (n_decoy_pulses_tick  * 0.5)
    yield_vacuum_obs = n_sifted_vacuum / (n_vacuum_pulses_tick  * 0.5 + 1e-10)

    Q_s = yield_signal_obs
    Q_d = yield_decoy_obs
    Y_0 = yield_vacuum_obs

    # Lo-Ma-Chen (2005) / Ma-Qi-Zhao-Lo (2005) "vacuum+weak decoy" bound.
    # FIXED: the original coefficient (MU_SIGNAL/(MU_SIGNAL-MU_DECOY)) and the
    # Q_signal term (Q_s*exp(MU_SIGNAL)/MU_SIGNAL) were missing required
    # scaling factors, causing y1_lower to evaluate NEGATIVE even under an
    # honest channel (clipping to exactly 0.0 for 100% of rows - verified
    # empirically). The correct standard formula (Ma, Qi, Zhao & Lo, Phys.
    # Rev. A 72, 012326 (2005), Eq. 11) is:
    #   Y_1^L = [mu / (nu*(mu - nu))] * ( Q_nu*e^nu - Q_mu*e^mu*(nu/mu)^2
    #            - ((mu^2 - nu^2)/mu^2) * Y_0 )
    # Verified this now gives y1_lower ~ eta_total under an honest channel
    # (physically correct: single-photon yield should be close to the
    # overall detection efficiency), instead of always clipping to zero.
    lmc_coeff = MU_SIGNAL / (MU_DECOY * (MU_SIGNAL - MU_DECOY))
    y1_lower = lmc_coeff * (
        Q_d * np.exp(MU_DECOY)
        - Q_s * np.exp(MU_SIGNAL) * (MU_DECOY / MU_SIGNAL) ** 2
        - ((MU_SIGNAL ** 2 - MU_DECOY ** 2) / MU_SIGNAL ** 2) * Y_0
    )
    y1_lower = np.clip(y1_lower, 0.0, None)

    # e1_upper: Ma-Qi-Zhao-Lo (2005) Eq. 12. Derivation: E_nu*Q_nu*e^nu =
    # sum_n (nu^n/n!) e_n Y_n = e_0*Y_0 + nu*e_1*Y_1 + (nonneg n>=2 terms),
    # with e_0 = 1/2 (vacuum error rate). The n=0 term carries NO nu factor,
    # so the numerator must subtract 0.5*Y_0, not 0.5*Y_0*MU_DECOY (fixed;
    # the previous version under-subtracted and inflated e1_upper).
    E_d = qber_obs_decoy
    e1_upper_num   = E_d * Q_d * np.exp(MU_DECOY) - 0.5 * Y_0
    e1_upper_denom = MU_DECOY * y1_lower + 1e-10
    e1_upper = np.clip(e1_upper_num / e1_upper_denom, 0.0, 0.5)

    # GLLP-style finite-key secure key rate: R = -Q_s*f_EC*H2(E_s) +
    # mu*e^{-mu}*Y1^L*(1-H2(e1_upper)). Q_s must multiply ONLY the
    # error-correction cost term (paid on every sifted bit); the
    # single-photon privacy-amplification term is already the single-photon
    # gain mu*e^{-mu}*Y1^L and must not be scaled by Q_s again (fixed; the
    # previous version multiplied both terms by Q_s, double-counting it in
    # the second term).
    E_s = qber_obs_signal
    secure_key_rate = (
        -Q_s * F_EC * _binary_entropy(E_s)
        + MU_SIGNAL * np.exp(-MU_SIGNAL) * y1_lower * (1.0 - _binary_entropy(e1_upper))
    )
    secure_key_rate = np.clip(secure_key_rate, 0.0, None)

    actual_yield_ratio = (Q_s + 1e-10) / (Q_d + 1e-10)
    decoy_anomaly = EXPECTED_YIELD_RATIO - actual_yield_ratio

    label = (kind_arr != "none").astype(int)

    df = pd.DataFrame({
        "run_id":                   run_id,
        "tick":                     t,
        "hardware":                 hardware_name,
        "run_type":                 run_type,
        "distance_km":              distance_km,
        "n_sifted":                 n_sifted,
        "observed_qber":            observed_qber,
        "attack_kind":              kind_arr,
        "attack_strength_instant":  strength_arr,
        "label":                    label,
        "yield_signal":             yield_signal_obs,
        "yield_decoy":              yield_decoy_obs,
        "yield_vacuum":             yield_vacuum_obs,
        "qber_signal":              qber_obs_signal,
        "qber_decoy":               qber_obs_decoy,
        "qber_delta":               qber_obs_signal - qber_obs_decoy,
        "yield_ratio":              actual_yield_ratio,
        "decoy_anomaly":            decoy_anomaly,
        "y1_lower":                 y1_lower,
        "e1_upper":                 e1_upper,
        "secure_key_rate":          secure_key_rate,
    })

    qber = df["observed_qber"]
    df["qber_roll_mean_30"]  = qber.rolling(30, min_periods=1).mean()
    df["qber_roll_std_30"]   = qber.rolling(30, min_periods=1).std().fillna(0)
    df["qber_slope_30"]      = qber.diff().rolling(30, min_periods=1).mean().fillna(0)
    df["sifted_rate_roll_mean_30"] = df["n_sifted"].rolling(30, min_periods=1).mean()
    df["qber_roll_mean_60"]  = qber.rolling(60, min_periods=1).mean()
    df["qber_roll_mean_120"] = qber.rolling(120, min_periods=1).mean()
    df["qber_roll_std_60"]   = qber.rolling(60, min_periods=1).std().fillna(0)
    df["qber_roll_std_120"]  = qber.rolling(120, min_periods=1).std().fillna(0)
    df["qber_cv_30"] = (df["qber_roll_std_30"] / (df["qber_roll_mean_30"] + 1e-8)).clip(0, 10)

    df["yield_ratio_roll_mean_30"]     = df["yield_ratio"].rolling(30, min_periods=1).mean()
    df["yield_ratio_roll_std_30"]      = df["yield_ratio"].rolling(30, min_periods=1).std().fillna(0)
    df["decoy_anomaly_roll_mean_30"]   = df["decoy_anomaly"].rolling(30, min_periods=1).mean()
    df["y1_lower_roll_mean_30"]        = df["y1_lower"].rolling(30, min_periods=1).mean()
    df["secure_key_rate_roll_mean_30"] = df["secure_key_rate"].rolling(30, min_periods=1).mean()
    df["secure_key_rate_roll_std_30"]  = df["secure_key_rate"].rolling(30, min_periods=1).std().fillna(0)

    for c in df.columns:
        if df[c].dtype == object and c not in ("hardware", "run_type", "attack_kind"):
            df[c] = df[c].astype(np.float64)

    return df


FEATURE_COLS_BASE = [
    "observed_qber", "qber_roll_mean_30", "qber_roll_std_30", "qber_slope_30",
    "sifted_rate_roll_mean_30", "n_sifted",
    "qber_roll_mean_60", "qber_roll_mean_120", "qber_roll_std_60",
    "qber_roll_std_120", "qber_cv_30",
]

NEW_DECOY_FEATURES = [
    "yield_signal", "yield_decoy", "yield_vacuum", "qber_signal", "qber_decoy",
    "qber_delta", "yield_ratio", "decoy_anomaly", "y1_lower", "e1_upper",
    "secure_key_rate", "yield_ratio_roll_mean_30", "yield_ratio_roll_std_30",
    "decoy_anomaly_roll_mean_30", "y1_lower_roll_mean_30",
    "secure_key_rate_roll_mean_30", "secure_key_rate_roll_std_30",
]

FEATURE_COLS = FEATURE_COLS_BASE + NEW_DECOY_FEATURES


def _stratified_run_types_by_distance(distances: np.ndarray, rng: np.random.Generator) -> list:
    n = len(distances)
    types = list(RUN_TYPE_PROBS.keys())
    target_fracs = np.array(list(RUN_TYPE_PROBS.values()))

    order = np.argsort(distances)
    assigned_counts = np.zeros(len(types))
    labels = np.empty(n, dtype=object)
    for step, pos in enumerate(order, start=1):
        deficit = target_fracs * step - assigned_counts
        choice = int(np.argmax(deficit))
        labels[pos] = types[choice]
        assigned_counts[choice] += 1

    n_swaps = max(1, n // 10)
    for _ in range(n_swaps):
        i = rng.integers(0, n - 1)
        pi, pj = order[i], order[i + 1]
        if labels[pi] != labels[pj]:
            labels[pi], labels[pj] = labels[pj], labels[pi]

    return labels.tolist()


def build_dataset(
    n_runs=100,
    duration_s=86_400,
    pulses_per_tick=200_000,
    seed=42,
    pns_qber_leakage_factor: float = PNS_QBER_LEAKAGE_FACTOR,
):
    rng = np.random.default_rng(seed)
    hardware_cycle = ["cooled", "uncooled"]
    n_cooled = n_runs // 2 + n_runs % 2
    n_uncooled = n_runs // 2

    cooled_distances = rng.uniform(5, 80, size=n_cooled)
    uncooled_distances = rng.uniform(5, 80, size=n_uncooled)

    cooled_types = _stratified_run_types_by_distance(cooled_distances, rng)
    uncooled_types = _stratified_run_types_by_distance(uncooled_distances, rng)

    cooled_idx = 0
    uncooled_idx = 0

    train_file = "qkd_train_decoy.csv"
    test_file = "qkd_test_decoy.csv"

    import os
    import gc

    if os.path.exists(train_file):
        os.remove(train_file)
    if os.path.exists(test_file):
        os.remove(test_file)

    train_header = True
    test_header = True

    for i in range(n_runs):
        run_rng = np.random.default_rng(seed * 1000 + i)
        hardware_name = hardware_cycle[i % 2]

        if hardware_name == "cooled":
            run_type = cooled_types[cooled_idx]
            distance_km = float(cooled_distances[cooled_idx])
            cooled_idx += 1
            output_file = train_file
            header = train_header
            train_header = False
        else:
            run_type = uncooled_types[uncooled_idx]
            distance_km = float(uncooled_distances[uncooled_idx])
            uncooled_idx += 1
            output_file = test_file
            header = test_header
            test_header = False

        df = simulate_long_timeline(
        i, hardware_name, run_type, duration_s, pulses_per_tick,
        distance_km=distance_km, rng=run_rng,
        pns_qber_leakage_factor=pns_qber_leakage_factor,
    )

        df.to_csv(output_file, mode="w" if header else "a", header=header, index=False)

        print(f"  run {i + 1}/{n_runs} done ({hardware_name}, {run_type}, dist={distance_km:.1f}km)")

        del df
        gc.collect()

    print("\nDataset generation complete.")
    print(f"Train: {train_file}")
    print(f"Test:  {test_file}")


def generalization_split(df: pd.DataFrame):
    runs = df[["run_id", "hardware"]].drop_duplicates()
    train_ids = runs[runs.hardware == "cooled"].run_id
    test_ids  = runs[~runs.run_id.isin(train_ids)].run_id
    train_df  = df[df.run_id.isin(train_ids)].reset_index(drop=True)
    test_df   = df[df.run_id.isin(test_ids)].reset_index(drop=True)
    return train_df, test_df


def report_run_type_counts(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("\n=== Run-type counts per split ===")
    for name, df in [("TRAIN", train_df), ("TEST", test_df)]:
        counts = df[["run_id", "run_type"]].drop_duplicates()["run_type"].value_counts()
        total = counts.sum()
        print(f"{name} ({total} runs): " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        if counts.min() < 10:
            print(f"  NOTE: smallest group has only {counts.min()} runs.")


def report_confound_diagnostic(train_df: pd.DataFrame) -> None:
    import itertools
    per_run = train_df.groupby(["run_id", "run_type"]).agg(
        distance_km=("distance_km", "first"), mean_n_sifted=("n_sifted", "mean")
    ).reset_index()
    summary = per_run.groupby("run_type").agg(
        distance_mean=("distance_km", "mean"), distance_std=("distance_km", "std"),
        sifted_mean=("mean_n_sifted", "mean"), sifted_std=("mean_n_sifted", "std"),
    )
    print("\n=== Confound diagnostic ===")
    for a, b in itertools.combinations(summary.index, 2):
        diff_d   = abs(summary.loc[a, "distance_mean"] - summary.loc[b, "distance_mean"])
        pooled_d = (summary.loc[a, "distance_std"] + summary.loc[b, "distance_std"]) / 2
        diff_s   = abs(summary.loc[a, "sifted_mean"] - summary.loc[b, "sifted_mean"])
        pooled_s = (summary.loc[a, "sifted_std"] + summary.loc[b, "sifted_std"]) / 2
        d_eff = diff_d / pooled_d if pooled_d > 0 else 0
        s_eff = diff_s / pooled_s if pooled_s > 0 else 0
        flag = "  <-- WARNING: confound" if max(d_eff, s_eff) > 0.3 else ""
        print(f"  {a} vs {b}: dist d={d_eff:.2f}, n_sifted d={s_eff:.2f}{flag}")


def validate_against_literature(df: pd.DataFrame) -> None:
    clean     = df[df.label == 0]
    ir_strong = df[(df.attack_kind == "intercept_resend") & (df.attack_strength_instant > 0.3)]
    pns_any   = df[df.attack_kind == "pns_like"]

    print("\n=== Literature sanity checks ===")
    print(f"Clean QBER mean:           {clean.observed_qber.mean():.4f}  (expect ~0.01-0.05)")
    print(f"Strong IR QBER mean:       {ir_strong.observed_qber.mean():.4f}  (expect >0.11)")
    print(f"PNS QBER mean:             {pns_any.observed_qber.mean():.4f}  (expect ~ clean)")
    print(f"Clean decoy_anomaly:       {clean.decoy_anomaly.mean():.4f}  (expect ~ 0 or slightly pos)")
    print(f"PNS decoy_anomaly:         {pns_any.decoy_anomaly.mean():.4f}  (expect > clean)")
    print(f"Clean y1_lower:            {clean.y1_lower.mean():.5f}")
    print(f"PNS y1_lower:              {pns_any.y1_lower.mean():.5f}  (expect << clean)")
    print(f"Clean yield_ratio:         {clean.yield_ratio.mean():.4f}  (expect ~ 5.0 absent dark counts)")
    print(f"PNS yield_ratio:           {pns_any.yield_ratio.mean():.4f}  (expect < clean)")

    floor_val = df["n_sifted"].min()
    frac_floor = (df["n_sifted"] == floor_val).mean()
    print(f"\nFinite-key: n_sifted floor={floor_val}, frac_at_floor={frac_floor:.3f} (ideally <0.05)")


def run_pulses_per_tick_ablation(
    n_runs: int = 10,
    duration_s: int = 1800,
    seed: int = 99,
    pulse_values: list = None,
) -> pd.DataFrame:
    """
    Diagnostic-only sweep. FIXED: build_dataset() now streams runs directly
    to CSV (train_file/test_file) rather than returning a concatenated
    DataFrame - a deliberate, good memory-efficiency change for the main
    864-run-scale generation. But this function still needs the generated
    data to compute diagnostics, so it now reads the written CSVs back in
    after each build_dataset() call, instead of trying to use a return
    value that no longer exists (the original version would have crashed
    with 'NoneType has no attribute label' the moment --ablation was used).
    """
    import os
    if pulse_values is None:
        pulse_values = [20_000, 50_000, 100_000, 200_000]
    rows = []
    for pv in pulse_values:
        print(f"\n--- pulses_per_tick = {pv:,} ---")
        build_dataset(n_runs=n_runs, duration_s=duration_s, pulses_per_tick=pv, seed=seed)
        df = pd.concat([pd.read_csv("qkd_train_decoy.csv"), pd.read_csv("qkd_test_decoy.csv")], ignore_index=True)
        clean = df[df.label == 0]
        floor_val = df["n_sifted"].min()
        frac_at_floor = (df["n_sifted"] == floor_val).mean()
        clean_cv = clean.observed_qber.std() / max(clean.observed_qber.mean(), 1e-12)
        rows.append(dict(
            pulses_per_tick=pv,
            n_sifted_mean=df["n_sifted"].mean(),
            frac_at_floor=frac_at_floor,
            clean_qber_mean=clean.observed_qber.mean(),
            clean_qber_cv=clean_cv,
        ))
    summary = pd.DataFrame(rows)
    print("\n=== pulses_per_tick ablation summary ===")
    print(summary.to_string(index=False))
    for f in ["qkd_train_decoy.csv", "qkd_test_decoy.csv"]:
        if os.path.exists(f):
            os.remove(f)  # these were ablation-scale test files, not your real dataset - clean up
    return summary

def run_qber_leakage_ablation(
    n_runs: int = 10,
    duration_s: int = 3600,
    seed: int = 77,
    leakage_values: list = None,
) -> pd.DataFrame:
    """Diagnostic-only. Checks whether PNS-vs-clean QBER separability
    depends heavily on the unvalidated PNS_QBER_LEAKAGE_FACTOR assumption."""
    import os
    from sklearn.metrics import roc_auc_score

    if leakage_values is None:
        leakage_values = [0.0, 0.01, 0.02, 0.05]

    rows = []
    for lf in leakage_values:
        print(f"\n--- leakage_factor = {lf} ---")
        build_dataset(n_runs=n_runs, duration_s=duration_s, seed=seed,
                      pns_qber_leakage_factor=lf)
        df = pd.concat([pd.read_csv("qkd_train_decoy.csv"),
                         pd.read_csv("qkd_test_decoy.csv")], ignore_index=True)
        mask = df.attack_kind.isin(["none", "pns_like"])
        y = (df.loc[mask, "attack_kind"] == "pns_like").astype(int)
        auc_raw = roc_auc_score(y, df.loc[mask, "observed_qber"])
        auc_roll = roc_auc_score(y, df.loc[mask, "qber_roll_mean_30"])
        rows.append(dict(leakage_factor=lf, auc_qber_raw=max(auc_raw, 1-auc_raw),
                          auc_qber_roll30=max(auc_roll, 1-auc_roll)))

    summary = pd.DataFrame(rows)
    print("\n=== PNS QBER-leakage sensitivity ===")
    print(summary.to_string(index=False))
    for f in ["qkd_train_decoy.csv", "qkd_test_decoy.csv"]:
        if os.path.exists(f):
            os.remove(f)
    return summary


if __name__ == "__main__":
    import sys
    import time

    if "--ablation" in sys.argv:
        run_pulses_per_tick_ablation()
        raise SystemExit(0)

    N_RUNS = 100
    start = time.time()
    print(f"Generating dataset ({N_RUNS} runs x 24h @ 1Hz, pulses_per_tick=200,000) with decoy-state features...")
    build_dataset(n_runs=N_RUNS, duration_s=86_400, pulses_per_tick=200_000, seed=42)
    elapsed = time.time() - start
    print(f"\nGeneration completed in {elapsed:.1f}s")
    print("Saved: qkd_train_decoy.csv")
    print("Saved: qkd_test_decoy.csv")

    # FIXED: these safety diagnostics existed in the file but were never
    # actually called here - meaning every check this project relied on to
    # catch real bugs earlier (finite-key floor, run-type balance, the
    # distance/throughput confound) was silently disabled for this version.
    # Re-enabled below, reading the just-written CSVs back in.
    print("\nRunning post-generation diagnostics...")
    train_df = pd.read_csv("qkd_train_decoy.csv")
    test_df = pd.read_csv("qkd_test_decoy.csv")
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    validate_against_literature(full_df)
    report_run_type_counts(train_df, test_df)
    report_confound_diagnostic(train_df)

    import contextlib
    with open("generation_diagnostics_log.txt", "w") as f, contextlib.redirect_stdout(f):
        validate_against_literature(full_df)
        report_run_type_counts(train_df, test_df)
        report_confound_diagnostic(train_df)
    print("Diagnostics saved to generation_diagnostics_log.txt — attach this file to the methodology writeup.")