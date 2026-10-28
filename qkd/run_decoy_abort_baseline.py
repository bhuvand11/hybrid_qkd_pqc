"""
run_decoy_abort_baseline.py
============================
Camera-ready addition: a genuine decoy-state key-rate/yield abort rule,
replacing the QBER-only strawman comparison Reviewer 2 flagged ("Deployed
decoy-state systems compute the single-photon yield bound Y1^L and the key
rate, and abort when the key rate is zero, not on QBER alone").

This is NOT a model -- it mirrors exactly how the existing "Static QBER
(per-link)" and "Combined static (QBER+throughput)" baselines in the main
pipeline (qkd_lstm_training_STEP4_0309.ipynb, section 11) are computed:
a per-run (per-link) robust z-score of a decoy-state feature, evaluated at
the same 60-tick/stride-30 windows (last-tick snapshot) as every other
method in Table 2 of the paper, with AUC-ROC/PR-AUC (threshold-free) as the
primary metric and F1 at each rule's own best threshold (same convention
the paper already uses for the other static rows).

Three decoy-state-only rules are reported:
  - Y1^L-abort:        suspicious when the single-photon yield bound drops
  - key-rate-abort:    suspicious when the secure key rate drops
  - decoy-state-abort: max of the two (mirrors how "Combined static"
                        already takes max(qber_z, -sifted_rate_z))

Runs entirely on the real, full-scale held-out test set already on disk
(qkd_test_decoy.csv, uncooled hardware, 4,320,000 rows) -- no GPU needed,
no Colab needed, this is pure feature engineering + AUC computation.
"""
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

TEST_CSV = "qkd_test_decoy.csv"
WINDOW_LEN = 60
STRIDE = 30

USECOLS = [
    "run_id", "tick", "attack_kind", "label",
    "y1_lower_roll_mean_30", "secure_key_rate_roll_mean_30",
]
DTYPES = {
    "run_id": "int32", "tick": "int32", "attack_kind": "category", "label": "int8",
    "y1_lower_roll_mean_30": "float32", "secure_key_rate_roll_mean_30": "float32",
}


def robust_normalize_per_run(df, feature_cols):
    df = df.sort_values(["run_id", "tick"]).reset_index(drop=True)
    med = df.groupby("run_id", observed=True)[feature_cols].transform("median")
    mad = (df[feature_cols] - med).abs()
    mad = mad.groupby(df["run_id"], observed=True).transform("median")
    scale = 1.4826 * mad + 1e-6
    z = (df[feature_cols] - med) / scale
    z = z.clip(-10, 10)
    for c in feature_cols:
        df[c + "_z"] = z[c].astype(np.float32)
    return df


def last_tick_per_window(df, feature_cols_z):
    """Same window construction as build_windows() in the main pipeline
    (60-tick windows, stride 30, never crossing a run boundary), but we only
    need the LAST tick's z-scored feature value per window -- exactly what
    the existing static-QBER / combined-static baselines use (the X_te_tab
    last-timestep snapshot)."""
    run_id = df["run_id"].values
    run_boundaries = np.flatnonzero(np.diff(run_id) != 0) + 1
    run_starts = np.concatenate([[0], run_boundaries])
    run_ends = np.concatenate([run_boundaries, [len(run_id)]])

    end_idx_list = []
    for s, e in zip(run_starts, run_ends):
        n = e - s
        if n < WINDOW_LEN:
            continue
        local_starts = np.arange(0, n - WINDOW_LEN + 1, STRIDE)
        end_idx_list.append(s + local_starts + WINDOW_LEN - 1)
    end_idx = np.concatenate(end_idx_list)

    out = {"label": df["label"].values[end_idx], "attack_kind": df["attack_kind"].values[end_idx]}
    for c in feature_cols_z:
        out[c] = df[c].values[end_idx]
    return pd.DataFrame(out)


def evaluate(score, y, attack_kind_arr):
    results = {}
    for name, ak_mask in [
        ("ALL", np.ones(len(y), dtype=bool)),
        ("PNS_only", np.isin(attack_kind_arr, ["none", "pns_like"])),
        ("IR_only", np.isin(attack_kind_arr, ["none", "intercept_resend"])),
    ]:
        yy, ss = y[ak_mask], score[ak_mask]
        if len(np.unique(yy)) < 2:
            results[name] = None
            continue
        auc = roc_auc_score(yy, ss)
        ap = average_precision_score(yy, ss)
        prec, rec, thr = precision_recall_curve(yy, ss)
        f1s = 2 * prec * rec / (prec + rec + 1e-12)
        f1 = float(np.nanmax(f1s[:-1])) if len(f1s) > 1 else 0.0
        results[name] = dict(auc=auc, ap=ap, f1=f1, n=len(yy), pos=int(yy.sum()))
    return results


def main():
    t0 = time.time()
    print(f"Reading {TEST_CSV} (columns: {USECOLS}) ...")
    df = pd.read_csv(TEST_CSV, usecols=USECOLS, dtype=DTYPES)
    print(f"  loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    feature_cols = ["y1_lower_roll_mean_30", "secure_key_rate_roll_mean_30"]
    df = robust_normalize_per_run(df, feature_cols)

    win = last_tick_per_window(df, [c + "_z" for c in feature_cols])
    print(f"  built {len(win):,} windows (60-tick, stride 30, last-tick snapshot)")

    y = win["label"].values.astype(np.int32)
    ak = win["attack_kind"].values

    y1_z = win["y1_lower_roll_mean_30_z"].values
    kr_z = win["secure_key_rate_roll_mean_30_z"].values

    # Lower Y1^L / lower key rate = more suspicious -> negate so higher score = more suspicious
    score_y1 = -y1_z
    score_keyrate = -kr_z
    score_decoy_abort = np.maximum(score_y1, score_keyrate)

    rows = []
    for rule_name, score in [
        ("Y1^L-abort (per-link)", score_y1),
        ("Key-rate-abort (per-link)", score_keyrate),
        ("Decoy-state abort (Y1^L + key-rate, per-link)", score_decoy_abort),
    ]:
        res = evaluate(score, y, ak)
        print(f"\n=== {rule_name} ===")
        for split, r in res.items():
            if r is None:
                print(f"  {split}: skipped (no positives)")
                continue
            print(f"  {split:10s} AUC={r['auc']:.4f}  PR-AUC={r['ap']:.4f}  F1={r['f1']:.4f}  "
                  f"n={r['n']}  pos={r['pos']}")
            rows.append(dict(rule=rule_name, split=split, auc=r["auc"], pr_auc=r["ap"],
                              f1=r["f1"], n=r["n"], pos=r["pos"]))

    out = pd.DataFrame(rows)
    out.to_csv("decoy_abort_baseline_results.csv", index=False)
    print(f"\nSaved decoy_abort_baseline_results.csv   (total time {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
