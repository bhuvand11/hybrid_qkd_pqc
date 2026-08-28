import os
import shutil
import pandas as pd
from sklearn.metrics import roc_auc_score

from generate_qkd_dataset import build_dataset

LEAKAGE_VALUES = [0.0, 0.01, 0.02, 0.05]   # the sensitivity range to test
SWEEP_DIR = "leakage_sweep_tmp"

os.makedirs(SWEEP_DIR, exist_ok=True)
orig_dir = os.getcwd()

results = []
try:
    os.chdir(SWEEP_DIR)   # isolate this run so it never touches your real CSVs
    for lf in LEAKAGE_VALUES:
        print(f"\n=== leakage_factor = {lf} ===")
        build_dataset(n_runs=20, duration_s=14_400, seed=77, pns_qber_leakage_factor=lf)

        df = pd.concat([
            pd.read_csv("qkd_train_decoy.csv"),
            pd.read_csv("qkd_test_decoy.csv"),
        ], ignore_index=True)

        n_pns_rows = (df["attack_kind"] == "pns_like").sum()
        print(f"  rows labeled pns_like: {n_pns_rows} / {len(df)}")

        mask = df["attack_kind"].isin(["none", "pns_like"])
        y = (df.loc[mask, "attack_kind"] == "pns_like").astype(int)

        if y.nunique() < 2:
            print(f"  WARNING: no PNS episodes landed at leakage={lf} -- "
                  f"duration_s still too short, skipping this value.")
            results.append(dict(leakage_factor=lf, auc_qber_raw=None, auc_qber_roll30=None))
            os.remove("qkd_train_decoy.csv")
            os.remove("qkd_test_decoy.csv")
            continue

        auc_raw  = roc_auc_score(y, df.loc[mask, "observed_qber"])
        auc_roll = roc_auc_score(y, df.loc[mask, "qber_roll_mean_30"])
        auc_raw  = max(auc_raw, 1 - auc_raw)
        auc_roll = max(auc_roll, 1 - auc_roll)

        results.append(dict(leakage_factor=lf, auc_qber_raw=auc_raw, auc_qber_roll30=auc_roll))
        print(f"  AUC(observed_qber) vs PNS      = {auc_raw:.4f}")
        print(f"  AUC(qber_roll_mean_30) vs PNS  = {auc_roll:.4f}")

        os.remove("qkd_train_decoy.csv")
        os.remove("qkd_test_decoy.csv")
finally:
    os.chdir(orig_dir)
    shutil.rmtree(SWEEP_DIR, ignore_errors=True)

summary = pd.DataFrame(results)
print("\n=== PNS QBER-leakage sensitivity summary ===")
print(summary.to_string(index=False))
summary.to_csv("leakage_sweep_results.csv", index=False)
print("\nSaved leakage_sweep_results.csv")