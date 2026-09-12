"""
Sanity-checks and re-does the fusion_baseline.py pipeline, fixing a
circularity bug found in the previous iteration's PQC ground-truth
definition. Every number below is either (a) verified against an
independent quantity, or (b) explicitly flagged as needing more work.
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.optimize import minimize_scalar
from scipy.special import logit as _logit

rng = np.random.default_rng(42)
QKD_DIR = r"d:\GitHub\hybrid_qkd_pqc\qkd"
PQC_DIR = r"d:\GitHub\hybrid_qkd_pqc\pqc\models\kyber_cnn_export"

print("=" * 70)
print("CHECK 1: Does Platt scaling preserve ranking (i.e. is it a valid,")
print("monotonic calibration, not something that scrambles the model)?")
print("=" * 70)

qkd_val_logits = np.load(f"{QKD_DIR}/qkd_val_logits.npy")
qkd_val_labels = np.load(f"{QKD_DIR}/qkd_val_labels.npy")
qkd_test_logits = np.load(f"{QKD_DIR}/qkd_test_logits.npy")
qkd_test_labels = np.load(f"{QKD_DIR}/qkd_test_labels.npy")
qkd_test_rid = np.load(f"{QKD_DIR}/qkd_test_run_ids.npy", allow_pickle=True)
qkd_test_ak = np.load(f"{QKD_DIR}/qkd_test_attack_kind.npy", allow_pickle=True)

platt = LogisticRegression()
platt.fit(qkd_val_logits.reshape(-1, 1), qkd_val_labels)
print(f"Platt coefficient sign: {platt.coef_[0][0]:+.4f} (must be positive -- "
      f"a negative coefficient would mean calibration inverted the model's ranking)")

raw_probs = 1 / (1 + np.exp(-qkd_test_logits))
qkd_test_calib = platt.predict_proba(qkd_test_logits.reshape(-1, 1))[:, 1]
auc_raw = roc_auc_score(qkd_test_labels, raw_probs)
auc_calib = roc_auc_score(qkd_test_labels, qkd_test_calib)
print(f"Window-level AUC BEFORE calibration (raw sigmoid): {auc_raw:.4f}")
print(f"Window-level AUC AFTER calibration (Platt):        {auc_calib:.4f}")
print(f"These should be near-identical (calibration is monotonic, AUC is rank-based) -- "
      f"difference = {abs(auc_raw-auc_calib):.6f}")
print("VERIFIED: matches the original notebook's cached test AUC of 0.9746 for the single-seed run." if abs(auc_raw - 0.9746) < 0.001 else
      "MISMATCH from expected 0.9746 -- investigate.")

print()
print("Calibration quality check (reliability): mean predicted prob vs actual")
print("positive rate, in 5 confidence bins -- these should track each other")
print("closely if calibration is doing its job.")
bins = np.quantile(qkd_test_calib, [0, .2, .4, .6, .8, 1.0])
for i in range(5):
    lo, hi = bins[i], bins[i + 1]
    mask = (qkd_test_calib >= lo) & (qkd_test_calib <= hi)
    if mask.sum() == 0:
        continue
    mean_pred = qkd_test_calib[mask].mean()
    actual_rate = qkd_test_labels[mask].mean()
    print(f"  bin [{lo:.3f},{hi:.3f}]  n={mask.sum():6d}  mean_predicted={mean_pred:.4f}  actual_positive_rate={actual_rate:.4f}")

print()
print("=" * 70)
print("CHECK 2: QKD session-level AUC=1.0000 -- is this real or a bug?")
print("=" * 70)

qkd_session_rows = []
for rid in np.unique(qkd_test_rid):
    mask = qkd_test_rid == rid
    probs = qkd_test_calib[mask]
    true_label = int(qkd_test_labels[mask].max())
    ak_vals = qkd_test_ak[mask]
    ak = str(ak_vals[ak_vals != "none"][0]) if (ak_vals != "none").any() else "none"
    qkd_session_rows.append(dict(run_id=int(rid), mean_prob=float(probs.mean()), qkd_true=true_label, attack_kind=ak))

clean_means = sorted(r["mean_prob"] for r in qkd_session_rows if r["qkd_true"] == 0)
attacked_means = sorted(r["mean_prob"] for r in qkd_session_rows if r["qkd_true"] == 1)
print(f"Clean sessions (n={len(clean_means)})    mean_prob range: [{min(clean_means):.4f}, {max(clean_means):.4f}]")
print(f"Attacked sessions (n={len(attacked_means)}) mean_prob range: [{min(attacked_means):.4f}, {max(attacked_means):.4f}]")
gap = min(attacked_means) - max(clean_means)
print(f"Gap between highest clean and lowest attacked: {gap:+.4f} "
      f"({'clean separation confirmed -- AUC=1.0 is real, not a bug' if gap > 0 else 'OVERLAP FOUND -- AUC=1.0 claim is WRONG, investigate'})")

report = {"qkd_calibration_check": {
    "platt_coef_sign_positive": bool(platt.coef_[0][0] > 0),
    "window_auc_before_calibration": round(float(auc_raw), 4),
    "window_auc_after_calibration": round(float(auc_calib), 4),
    "matches_original_cached_result_0_9746": bool(abs(auc_raw - 0.9746) < 0.001),
}, "qkd_session_separation_check": {
    "clean_session_mean_prob_range": [round(min(clean_means), 4), round(max(clean_means), 4)],
    "attacked_session_mean_prob_range": [round(min(attacked_means), 4), round(max(attacked_means), 4)],
    "gap": round(float(gap), 4),
    "auc_1_0_is_genuine": bool(gap > 0),
}}

print()
print("=" * 70)
print("CHECK 3: The PQC circularity bug -- confirming it, then fixing it")
print("=" * 70)

pqc_val_probs = np.load(f"{PQC_DIR}/pqc_val_probs.npy")
pqc_val_labels = np.load(f"{PQC_DIR}/pqc_val_labels.npy")
pqc_test_probs = np.load(f"{PQC_DIR}/pqc_test_probs.npy")
pqc_test_labels = np.load(f"{PQC_DIR}/pqc_test_labels.npy")

eps = 1e-12
val_log_probs = np.log(pqc_val_probs + eps)


def nll_at_temperature(T, log_probs, labels):
    scaled = log_probs / T
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_ = np.exp(scaled)
    softmax = exp_ / exp_.sum(axis=1, keepdims=True)
    n = len(labels)
    return -np.mean(np.log(softmax[np.arange(n), labels] + eps))


res = minimize_scalar(nll_at_temperature, bounds=(0.05, 10.0), method="bounded", args=(val_log_probs, pqc_val_labels))
T_star = res.x

test_log_probs = np.log(pqc_test_probs + eps)
scaled = test_log_probs / T_star
scaled -= scaled.max(axis=1, keepdims=True)
exp_ = np.exp(scaled)
pqc_test_calib_probs = exp_ / exp_.sum(axis=1, keepdims=True)
pqc_pred_class = np.argmax(pqc_test_calib_probs, axis=1)
pqc_conf = pqc_test_calib_probs[np.arange(len(pqc_pred_class)), pqc_pred_class]
pqc_correct = (pqc_pred_class == pqc_test_labels)

print("Previous iteration's definition: compromised = correct AND confidence > median(confidence)")
print("Problem: 'confidence > median' is computed FROM the same variable (pqc_conf) used as the")
print("predictor being evaluated. Any predictor thresholded-and-then-evaluated-against-itself will")
print("look artificially strong -- this is circular, not a genuine finding.")
print()
print("PROPER FIX: use the honest, real ground truth (raw top-1 correctness, no manipulation of")
print("the label using the predictor itself). Accept that this is genuinely ~95.6% positive --")
print("that's a true fact about this specific known-unprotected reference device, not a metric")
print("artifact. Report AUC (rank-based, insensitive to base rate) as the primary number, and")
print("do not lean on PR-AUC for this branch since it is base-rate-sensitive by construction.")

pqc_true_honest = pqc_correct.astype(int)
print(f"\nHonest positive rate: {pqc_true_honest.mean()*100:.2f}% (matches original CNN test accuracy 95.64%: "
      f"{'YES' if abs(pqc_true_honest.mean() - 0.9564) < 0.001 else 'NO -- investigate'})")

auc_honest = roc_auc_score(pqc_true_honest, pqc_conf)
print(f"Trace-level AUC (calibrated confidence predicting HONEST correctness): {auc_honest:.4f}")

report["pqc_circularity_fix"] = {
    "bug_confirmed": True,
    "bug_description": "previous ground truth (confidence > median) was derived from the same variable used as the predictor, artificially inflating AUC",
    "fix": "reverted to raw top-1 correctness as ground truth -- no manipulation",
    "honest_positive_rate": round(float(pqc_true_honest.mean()), 4),
    "matches_original_test_accuracy_0_9564": bool(abs(pqc_true_honest.mean() - 0.9564) < 0.001),
    "trace_auc_honest": round(float(auc_honest), 4),
}

print()
print("=" * 70)
print("CHECK 4: Re-run the joint fusion evaluation with the HONEST PQC label")
print("=" * 70)
print("Base rate will be back to ~95.6% for PQC, so PR-AUC will sit near a high")
print("ceiling for every detector -- that is an accurate reflection of reality")
print("(this device is genuinely almost always exploitable), not a flaw. AUC")
print("remains the trustworthy number here since it does not depend on base rate.")

N_PAIRS = 20000
qkd_idx = rng.integers(0, len(qkd_session_rows), size=N_PAIRS)
pqc_idx = rng.integers(0, len(pqc_conf), size=N_PAIRS)

qkd_risk_paired = np.array([qkd_session_rows[i]["mean_prob"] for i in qkd_idx])
qkd_true_paired = np.array([qkd_session_rows[i]["qkd_true"] for i in qkd_idx])
pqc_risk_paired = pqc_conf[pqc_idx]
pqc_true_paired = pqc_true_honest[pqc_idx]

joint_true = np.maximum(qkd_true_paired, pqc_true_paired)
print(f"\nJoint positive rate: {joint_true.mean()*100:.2f}%")


def logit_clip(p, eps=1e-6):
    return _logit(np.clip(p, eps, 1 - eps))


fused_score = 1 / (1 + np.exp(-(logit_clip(qkd_risk_paired) + logit_clip(pqc_risk_paired))))

results_table = {}
for name, score in [("qkd_alone", qkd_risk_paired), ("pqc_alone", pqc_risk_paired), ("fused_log_odds", fused_score)]:
    auc = roc_auc_score(joint_true, score)
    ap = average_precision_score(joint_true, score)
    results_table[name] = {"auc": round(float(auc), 4), "pr_auc": round(float(ap), 4)}

print(f"\n{'Detector':<20}{'AUC':>10}{'PR-AUC':>10}   (PR-AUC not meaningful here -- see note above)")
for name, vals in results_table.items():
    print(f"{name:<20}{vals['auc']:>10.4f}{vals['pr_auc']:>10.4f}")

# Supplementary, base-rate-robust metric: sensitivity (recall) at a fixed,
# low false-positive-rate operating point, chosen from the ROC curve itself
# (5% FPR -- a standard, commonly reported IDS operating point), not picked
# to make any particular method look good.
from sklearn.metrics import roc_curve
print("\nSupplementary check -- recall at a fixed 5% false-positive-rate operating point")
print("(chosen because AUC alone can look similar even when operating behavior differs):")
for name, score in [("qkd_alone", qkd_risk_paired), ("pqc_alone", pqc_risk_paired), ("fused_log_odds", fused_score)]:
    fpr, tpr, thr = roc_curve(joint_true, score)
    idx = np.searchsorted(fpr, 0.05)
    idx = min(idx, len(tpr) - 1)
    print(f"  {name:<20} recall at ~5% FPR: {tpr[idx]:.4f}")
    results_table[name]["recall_at_5pct_fpr"] = round(float(tpr[idx]), 4)

report["joint_evaluation_honest"] = {
    "n_bootstrap_pairs": N_PAIRS,
    "joint_positive_rate": round(float(joint_true.mean()), 4),
    "note_on_pr_auc": "PR-AUC is near-ceiling for all detectors here because the true PQC base rate (95.6%) is genuinely high -- this reflects that the reference device is almost always exploitable, not a flaw in the test. AUC and the 5%-FPR recall figure below are the metrics to trust.",
    "results": results_table,
}

OUT_PATH = r"d:\GitHub\hybrid_qkd_pqc\fusion\sanity_check_results.json"
with open(OUT_PATH, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nSaved full sanity-check report to {OUT_PATH}")
