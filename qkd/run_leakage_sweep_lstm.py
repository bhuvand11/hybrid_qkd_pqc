"""
Extended PNS-QBER-leakage sensitivity sweep: Tier 2 of the lambda check.

run_leakage_sweep.py (the original sweep behind Table 10) only re-measures
the STATIC-QBER baseline across lambda in {0, 0.01, 0.02, 0.05}. It does not
re-measure the LSTM, because Y1_L/decoy_anomaly are provably lambda-invariant
by construction (see generate_qkd_dataset.py: lambda only enters
`pns_contrib`, which only feeds `true_qber`/`true_qber_decoy` -- it never
touches Q_s/Q_d/Y_0, which is all Y1_L and decoy_anomaly are computed from).
That structural argument covers the decoy-state-feature side for free.

It does NOT automatically cover the LSTM, because several of its 9 input
features ARE QBER-derived (observed_qber, qber_roll_*, and
secure_key_rate_roll_mean_30 indirectly via e1_upper). This script closes
that remaining gap empirically: for each lambda, it rebuilds the SAME
20-run/4-hour diagnostic dataset run_leakage_sweep.py uses (apples-to-apples
with the existing Table 10 numbers), trains a fresh LSTM on it with the
exact architecture/hyperparameters from the main pipeline
(qkd_lstm_training_STEP4_0309.ipynb), and reports its PNS/IR/ALL AUC
alongside the already-known QBER numbers.

Needs a GPU (Colab T4 is fine) -- dataset generation is CPU-only and fast,
but LSTM training wants CUDA. Falls back to CPU automatically if none is
found, just slower.
"""
import os
import time
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from generate_qkd_dataset import build_dataset

# ---------------------------------------------------------------------------
# Config -- mirrors run_leakage_sweep.py (same n_runs/duration_s/seed, so the
# QBER numbers reproduce exactly) and qkd_lstm_training_STEP4_0309.ipynb
# (same features/window/model/training hyperparameters, so the LSTM number
# at lambda=0.02 is a fair diagnostic-scale reference point).
# ---------------------------------------------------------------------------
LEAKAGE_VALUES = [0.0, 0.01, 0.02, 0.05]
SWEEP_DIR = "leakage_sweep_lstm_tmp"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: no GPU detected -- this will be slow. In Colab: "
          "Runtime -> Change runtime type -> T4 GPU, then re-run.")

FEATURE_COLS = [
    "observed_qber", "qber_roll_mean_30", "qber_roll_std_30", "qber_cv_30", "qber_slope_30",
    "decoy_anomaly_roll_mean_30", "y1_lower_roll_mean_30",
    "secure_key_rate_roll_mean_30", "sifted_rate_roll_mean_30",
]
Z_COLS = [c + "_z" for c in FEATURE_COLS]
WINDOW_LEN = 60
STRIDE = 30
BATCH_SIZE = 256
N_EPOCHS = 15
PATIENCE = 3
LR = 1e-3


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


def build_windows(df, feature_cols, window_len, stride):
    feat = df[feature_cols].values.astype(np.float32)
    label = df["label"].values.astype(np.float32)
    attack_kind = df["attack_kind"].values
    run_id = df["run_id"].values

    run_boundaries = np.flatnonzero(np.diff(run_id) != 0) + 1
    run_starts = np.concatenate([[0], run_boundaries])
    run_ends = np.concatenate([run_boundaries, [len(run_id)]])

    starts_list = []
    for s, e in zip(run_starts, run_ends):
        n = e - s
        if n < window_len:
            continue
        local_starts = np.arange(0, n - window_len + 1, stride)
        starts_list.append(s + local_starts)
    starts = np.concatenate(starts_list)
    ends = starts + window_len
    win_labels = label[ends - 1]
    win_attack_kind = attack_kind[ends - 1]
    return feat, starts, ends, win_labels, win_attack_kind


class QKDWindowDataset(Dataset):
    def __init__(self, feat, starts, ends, labels):
        self.feat, self.starts, self.ends, self.labels = feat, starts, ends, labels

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s, e = self.starts[idx], self.ends[idx]
        return torch.from_numpy(self.feat[s:e]), torch.tensor(self.labels[idx], dtype=torch.float32)


class LSTMDetector(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


def run_eval_loader(model, loader):
    model.eval()
    logits, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            logits.append(model(xb).float().cpu().numpy())
            ys.append(yb.numpy())
    return np.concatenate(logits), np.concatenate(ys)


def evaluate_probs(probs, y, attack_kind_arr, threshold=0.5):
    results = {}
    for name, ak_mask in [
        ("ALL", np.ones(len(y), dtype=bool)),
        ("PNS_only", np.isin(attack_kind_arr, ["none", "pns_like"])),
        ("IR_only", np.isin(attack_kind_arr, ["none", "intercept_resend"])),
    ]:
        yy, pp = y[ak_mask], probs[ak_mask]
        if len(np.unique(yy)) < 2:
            results[name] = None
            continue
        results[name] = dict(
            auc=roc_auc_score(yy, pp),
            ap=average_precision_score(yy, pp),
            f1=f1_score(yy, (pp > threshold).astype(int)),
            n=len(yy), pos=int(yy.sum()),
        )
    return results


def train_lstm_on(train_df, test_df):
    train_df = robust_normalize_per_run(train_df, FEATURE_COLS)
    test_df = robust_normalize_per_run(test_df, FEATURE_COLS)

    train_feat, train_starts, train_ends, train_labels, train_ak = build_windows(
        train_df, Z_COLS, WINDOW_LEN, STRIDE)
    test_feat, test_starts, test_ends, test_labels, test_ak = build_windows(
        test_df, Z_COLS, WINDOW_LEN, STRIDE)
    train_rid = train_df["run_id"].values[train_ends - 1]

    print(f"  train windows: {len(train_starts)}   test windows: {len(test_starts)}")
    if len(np.unique(train_labels)) < 2 or len(np.unique(test_labels)) < 2:
        print("  WARNING: not enough positive/negative windows at this scale -- skipping LSTM.")
        return None

    rng = np.random.default_rng(SEED)
    train_run_ids = np.unique(train_rid)
    n_val = max(1, len(train_run_ids) // 5)
    val_run_ids = rng.choice(train_run_ids, size=n_val, replace=False)
    is_val = np.isin(train_rid, val_run_ids)

    tr_ds = QKDWindowDataset(train_feat, train_starts[~is_val], train_ends[~is_val], train_labels[~is_val])
    va_ds = QKDWindowDataset(train_feat, train_starts[is_val], train_ends[is_val], train_labels[is_val])
    te_ds = QKDWindowDataset(test_feat, test_starts, test_ends, test_labels)

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=512, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=512, shuffle=False)

    model = LSTMDetector(n_features=len(Z_COLS)).to(DEVICE)
    pos_frac = train_labels[~is_val].mean()
    pos_weight = torch.tensor([(1 - pos_frac) / max(pos_frac, 1e-6)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_val_auc, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(N_EPOCHS):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            with torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        if len(va_ds) == 0:
            val_auc = float("nan")
        else:
            val_logits, val_y = run_eval_loader(model, va_loader)
            val_auc = roc_auc_score(val_y, val_logits) if len(np.unique(val_y)) > 1 else float("nan")
        if not np.isnan(val_auc):
            scheduler.step(val_auc)
        print(f"    epoch {epoch+1:2d}/{N_EPOCHS}  val_auc={val_auc:.4f}")

        score = val_auc if not np.isnan(val_auc) else epoch  # fallback: last epoch if val too small
        if score > best_val_auc:
            best_val_auc = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"    early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_logits, test_y = run_eval_loader(model, te_loader)
    test_probs = 1 / (1 + np.exp(-test_logits))
    return evaluate_probs(test_probs, test_y, test_ak)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
os.makedirs(SWEEP_DIR, exist_ok=True)
orig_dir = os.getcwd()
results = []
try:
    os.chdir(SWEEP_DIR)
    for lf in LEAKAGE_VALUES:
        print(f"\n=== leakage_factor (lambda) = {lf} ===")
        t0 = time.time()
        build_dataset(n_runs=20, duration_s=14_400, seed=77, pns_qber_leakage_factor=lf)
        print(f"  dataset generated in {time.time()-t0:.1f}s")

        train_df = pd.read_csv("qkd_train_decoy.csv")
        test_df = pd.read_csv("qkd_test_decoy.csv")
        full_df = pd.concat([train_df, test_df], ignore_index=True)

        n_pns_rows = (full_df["attack_kind"] == "pns_like").sum()
        print(f"  rows labeled pns_like: {n_pns_rows} / {len(full_df)}")

        mask = full_df["attack_kind"].isin(["none", "pns_like"])
        y_qber = (full_df.loc[mask, "attack_kind"] == "pns_like").astype(int)

        row = dict(leakage_factor=lf)
        if y_qber.nunique() < 2:
            print("  WARNING: no PNS episodes at this leakage value -- skipping QBER+LSTM.")
            row.update(auc_qber_raw=None, auc_qber_roll30=None,
                        auc_lstm_pns=None, auc_lstm_all=None, auc_lstm_ir=None)
        else:
            auc_raw = roc_auc_score(y_qber, full_df.loc[mask, "observed_qber"])
            auc_roll = roc_auc_score(y_qber, full_df.loc[mask, "qber_roll_mean_30"])
            row["auc_qber_raw"] = max(auc_raw, 1 - auc_raw)
            row["auc_qber_roll30"] = max(auc_roll, 1 - auc_roll)
            print(f"  AUC(observed_qber) vs PNS      = {row['auc_qber_raw']:.4f}")
            print(f"  AUC(qber_roll_mean_30) vs PNS  = {row['auc_qber_roll30']:.4f}")

            print("  training LSTM on this diagnostic-scale dataset...")
            t0 = time.time()
            lstm_res = train_lstm_on(train_df.copy(), test_df.copy())
            print(f"  LSTM done in {time.time()-t0:.1f}s")
            if lstm_res is not None:
                row["auc_lstm_pns"] = lstm_res["PNS_only"]["auc"] if lstm_res["PNS_only"] else None
                row["auc_lstm_all"] = lstm_res["ALL"]["auc"] if lstm_res["ALL"] else None
                row["auc_lstm_ir"] = lstm_res["IR_only"]["auc"] if lstm_res["IR_only"] else None
                print(f"  LSTM AUC vs PNS = {row['auc_lstm_pns']:.4f}   "
                      f"vs ALL = {row['auc_lstm_all']:.4f}   vs IR = {row['auc_lstm_ir']:.4f}")
            else:
                row.update(auc_lstm_pns=None, auc_lstm_all=None, auc_lstm_ir=None)

        results.append(row)
        os.remove("qkd_train_decoy.csv")
        os.remove("qkd_test_decoy.csv")
finally:
    os.chdir(orig_dir)
    shutil.rmtree(SWEEP_DIR, ignore_errors=True)

summary = pd.DataFrame(results)
print("\n=== PNS QBER-leakage sensitivity summary (QBER baseline + LSTM) ===")
print(summary.to_string(index=False))
summary.to_csv("leakage_sweep_lstm_results.csv", index=False)
print("\nSaved leakage_sweep_lstm_results.csv -- download this file, that's the only output you need.")
