"""
Fusion v2: corrected architecture.

v1's mistake: treated PQC's per-trace "was the classification correct"
signal as if it were a live, per-session random variable to be paired
and combined with QKD's real-time detector. That's a category error --
per-trace correctness can only be checked against the real secret, which
a real deployment never has in real time. It's a one-time LAB
CERTIFICATION metric about the hardware, not a session-level signal.

v2: QKD stays a real-time per-session detector (unchanged, already
validated). PQC becomes a fixed device-level risk floor (with a proper
binomial confidence interval, since it's estimated from a finite test
set) that shifts the ABSOLUTE trust level of every session from that
hardware, without being expected to help distinguish WHICH session was
attacked (that's still purely QKD's job, and rightly so).
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.special import logit as _logit

QKD_DIR = r"d:\GitHub\hybrid_qkd_pqc\qkd"
PQC_DIR = r"d:\GitHub\hybrid_qkd_pqc\pqc\models\kyber_cnn_export"
OUT_PATH = r"d:\GitHub\hybrid_qkd_pqc\fusion\fusion_v2_results.json"

# ============================================================
# 1. QKD: unchanged from the validated v1 pipeline (Platt-calibrated,
#    mean-probability session aggregation). Re-derived here for a
#    self-contained script.
# ============================================================
qkd_val_logits = np.load(f"{QKD_DIR}/qkd_val_logits.npy")
qkd_val_labels = np.load(f"{QKD_DIR}/qkd_val_labels.npy")
qkd_test_logits = np.load(f"{QKD_DIR}/qkd_test_logits.npy")
qkd_test_labels = np.load(f"{QKD_DIR}/qkd_test_labels.npy")
qkd_test_rid = np.load(f"{QKD_DIR}/qkd_test_run_ids.npy", allow_pickle=True)
qkd_test_ak = np.load(f"{QKD_DIR}/qkd_test_attack_kind.npy", allow_pickle=True)

platt = LogisticRegression()
platt.fit(qkd_val_logits.reshape(-1, 1), qkd_val_labels)
qkd_test_calib = platt.predict_proba(qkd_test_logits.reshape(-1, 1))[:, 1]

sessions = []
for rid in np.unique(qkd_test_rid):
    mask = qkd_test_rid == rid
    probs = qkd_test_calib[mask]
    true_label = int(qkd_test_labels[mask].max())
    ak_vals = qkd_test_ak[mask]
    ak = str(ak_vals[ak_vals != "none"][0]) if (ak_vals != "none").any() else "none"
    sessions.append(dict(run_id=int(rid), qkd_risk=float(probs.mean()), qkd_true=true_label, attack_kind=ak))

qkd_ranking_auc = roc_auc_score([s["qkd_true"] for s in sessions], [s["qkd_risk"] for s in sessions])
print(f"[QKD] session-detection AUC (unchanged from v1): {qkd_ranking_auc:.4f}")

# ============================================================
# 2. PQC: device-level risk floor with a real confidence interval,
#    not a per-session random draw.
# ============================================================
pqc_test_probs = np.load(f"{PQC_DIR}/pqc_test_probs.npy")
pqc_test_labels = np.load(f"{PQC_DIR}/pqc_test_labels.npy")
pqc_pred = np.argmax(pqc_test_probs, axis=1)
n_correct = int((pqc_pred == pqc_test_labels).sum())
n_total = len(pqc_test_labels)
p_hat = n_correct / n_total

# Wilson score interval (95%) -- the standard, correct way to put an
# honest uncertainty band on a proportion estimated from a finite sample,
# rather than reporting 95.64% as if it were exact.
z = 1.96
denom = 1 + z**2 / n_total
center = (p_hat + z**2 / (2 * n_total)) / denom
halfwidth = (z * np.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2))) / denom
ci_lo, ci_hi = center - halfwidth, center + halfwidth

print(f"[PQC] device leak rate (our reference implementation): {p_hat:.4f}  "
      f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]  (n={n_total} traces)")

# Comparison scenario -- reuses the ALREADY-VALIDATED classical PCA+RF
# baseline (pqc/models/kyber_cnn_export/baseline_pca_rf_results.json) as
# a stand-in for "a harder-to-attack implementation", so the framework's
# behavior can be shown under two real, already-measured risk levels
# instead of inventing a fake number for contrast.
with open(f"{PQC_DIR}/baseline_pca_rf_results.json") as f:
    rf_baseline = json.load(f)
p_hat_safer = rf_baseline["test_accuracy_pct"] / 100.0
n_safer = rf_baseline["test_set_size"]
center_s = (p_hat_safer + z**2 / (2 * n_safer)) / (1 + z**2 / n_safer)
halfwidth_s = (z * np.sqrt(p_hat_safer * (1 - p_hat_safer) / n_safer + z**2 / (4 * n_safer**2))) / (1 + z**2 / n_safer)
ci_lo_s, ci_hi_s = center_s - halfwidth_s, center_s + halfwidth_s
print(f"[PQC] device leak rate (classical-baseline stand-in for a harder target): {p_hat_safer:.4f}  "
      f"95% CI [{ci_lo_s:.4f}, {ci_hi_s:.4f}]  (n={n_safer} traces)")

# ============================================================
# 3. Fuse: QKD's live per-session log-odds + PQC's fixed device log-odds
# ============================================================
def logit_clip(p, eps=1e-6):
    return float(_logit(np.clip(p, eps, 1 - eps)))

pqc_logodds_leaky = logit_clip(p_hat)
pqc_logodds_safer = logit_clip(p_hat_safer)

for s in sessions:
    qkd_lo = logit_clip(s["qkd_risk"])
    s["fused_trust_leaky_device"] = float(1 / (1 + np.exp(-(qkd_lo + pqc_logodds_leaky))))
    s["fused_trust_safer_device"] = float(1 / (1 + np.exp(-(qkd_lo + pqc_logodds_safer))))
    # trust = 1 - compromise probability, easier to read
    s["trust_score_leaky_device_pct"] = round((1 - s["fused_trust_leaky_device"]) * 100, 2)
    s["trust_score_safer_device_pct"] = round((1 - s["fused_trust_safer_device"]) * 100, 2)

# Confirm ranking is UNCHANGED by fusion (expected and correct: a constant
# per-device shift cannot change which sessions look more/less suspicious
# relative to each other -- fusion's job here is calibrating the absolute
# level, not re-ranking sessions).
fused_ranking_auc = roc_auc_score(
    [s["qkd_true"] for s in sessions],
    [s["fused_trust_leaky_device"] for s in sessions],
)
print(f"\n[CHECK] session-detection AUC after fusion: {fused_ranking_auc:.4f} "
      f"(should equal QKD-alone {qkd_ranking_auc:.4f} exactly -- {'MATCHES, as expected' if abs(fused_ranking_auc-qkd_ranking_auc)<1e-9 else 'MISMATCH -- bug'})")

# Concrete, readable example: cleanest-looking session and the
# most-attacked session, under both device scenarios.
clean_example = min(sessions, key=lambda s: s["qkd_risk"])
attacked_example = max(sessions, key=lambda s: s["qkd_risk"])

print("\n=== Concrete example: same QKD session, two different PQC hardware ===")
for label, ex in [("Cleanest-looking QKD session", clean_example), ("Most-attacked QKD session", attacked_example)]:
    print(f"\n{label} (run_id={ex['run_id']}, attack_kind={ex['attack_kind']}, QKD raw risk={ex['qkd_risk']:.4f}):")
    print(f"  Trust score if PQC hardware = our leaky reference chip:      {ex['trust_score_leaky_device_pct']}%")
    print(f"  Trust score if PQC hardware = classical-baseline stand-in:   {ex['trust_score_safer_device_pct']}%")

report = {
    "qkd_session_detection_auc": round(float(qkd_ranking_auc), 4),
    "pqc_device_risk": {
        "leaky_reference_device": {"point_estimate": round(p_hat, 4), "wilson_95ci": [round(ci_lo, 4), round(ci_hi, 4)], "n": n_total},
        "safer_baseline_stand_in": {"point_estimate": round(p_hat_safer, 4), "wilson_95ci": [round(ci_lo_s, 4), round(ci_hi_s, 4)], "n": n_safer},
    },
    "fused_ranking_auc_matches_qkd_alone": bool(abs(fused_ranking_auc - qkd_ranking_auc) < 1e-9),
    "design_note": "Fusion is not expected to change which sessions rank as suspicious (that's QKD's job) -- it correctly calibrates the ABSOLUTE trust level based on known hardware risk, which QKD alone cannot see.",
    "all_sessions": sessions,
    "concrete_examples": {
        "cleanest_session": {k: v for k, v in clean_example.items()},
        "most_attacked_session": {k: v for k, v in attacked_example.items()},
    },
}
with open(OUT_PATH, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nSaved full v2 report to {OUT_PATH}")
