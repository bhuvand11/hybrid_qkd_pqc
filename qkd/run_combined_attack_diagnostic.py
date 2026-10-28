"""
run_combined_attack_diagnostic.py
==================================
Two camera-ready additions, run together since both reuse the same trained
LSTM:

1. COMBINED-ATTACK ROBUSTNESS. generate_qkd_dataset.py's simulate_long_timeline
   now supports run_type="combined" (IR and PNS episodes sampled independently
   over the same timeline, so they can co-occur on the same ticks, unlike
   "pure_ir"/"pure_pns" which are mutually exclusive by construction). This
   script builds a small diagnostic dataset with four run types -- clean,
   pure_ir, pure_pns, combined -- trains the same LSTM architecture/features/
   windowing used throughout the main pipeline (mirrors run_leakage_sweep_lstm.py),
   and reports AUC-ROC/PR-AUC/F1 broken out by attack type, including the new
   "combined_only" slice.

2. INFERENCE LATENCY. Using that same trained model, measures per-window
   LSTM inference latency (single-window forward pass, batch size 1) so the
   paper can state a concrete real-time-budget number instead of only
   asserting the pipeline is "real-time" in the abstract.

Mirrors run_leakage_sweep_lstm.py's architecture/hyperparameters exactly so
the combined-attack AUC is a fair, apples-to-apples diagnostic-scale number
against the existing pure_ir/pure_pns results already in the paper.

Needs a GPU for speed (Colab T4 is fine); falls back to CPU automatically.
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

from generate_qkd_dataset import simulate_long_timeline

# ---------------------------------------------------------------------------
# Config -- same features/window/model/training hyperparameters as
# qkd_lstm_training_STEP4_0309.ipynb / run_leakage_sweep_lstm.py, so results
# are directly comparable to the existing pure_ir/pure_pns diagnostic numbers.
# ---------------------------------------------------------------------------
DIAG_DIR = "combined_attack_diagnostic_tmp"
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

# Diagnostic-scale dataset: same duration/hardware split pattern as the main
# pipeline (cooled trains, uncooled tests), but small enough to run quickly.
N_RUNS_PER_TYPE = 6          # -> 24 runs total (6 x 4 run types), half cooled/half uncooled
DURATION_S = 14_400          # 4 hours, same scale as run_leakage_sweep_lstm.py
RUN_TYPES = ["clean", "pure_ir", "pure_pns", "combined"]


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
        ("combined_only", np.isin(attack_kind_arr, ["none", "combined"])),
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


def build_diagnostic_dataset():
    """Generates a small clean/pure_ir/pure_pns/combined dataset, split by
    hardware the same way the main pipeline is (cooled trains, uncooled
    tests). Returns (train_df, test_df)."""
    rows_train, rows_test = [], []
    run_id = 0
    for run_type in RUN_TYPES:
        for i in range(N_RUNS_PER_TYPE):
            hardware_name = "cooled" if i % 2 == 0 else "uncooled"
            run_rng = np.random.default_rng(SEED * 1000 + run_id)
            df = simulate_long_timeline(
                run_id, hardware_name, run_type, duration_s=DURATION_S,
                pulses_per_tick=200_000, rng=run_rng,
            )
            (rows_train if hardware_name == "cooled" else rows_test).append(df)
            run_id += 1
    train_df = pd.concat(rows_train, ignore_index=True)
    test_df = pd.concat(rows_test, ignore_index=True)
    return train_df, test_df


def train_lstm_on(train_df, test_df):
    train_df = robust_normalize_per_run(train_df, FEATURE_COLS)
    test_df = robust_normalize_per_run(test_df, FEATURE_COLS)

    train_feat, train_starts, train_ends, train_labels, train_ak = build_windows(
        train_df, Z_COLS, WINDOW_LEN, STRIDE)
    test_feat, test_starts, test_ends, test_labels, test_ak = build_windows(
        test_df, Z_COLS, WINDOW_LEN, STRIDE)
    train_rid = train_df["run_id"].values[train_ends - 1]

    print(f"  train windows: {len(train_starts)}   test windows: {len(test_starts)}")

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

        score = val_auc if not np.isnan(val_auc) else epoch
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
    results = evaluate_probs(test_probs, test_y, test_ak)

    # --- Inference latency: single-window (batch size 1) forward pass ---
    model.eval()
    n_feat = len(Z_COLS)
    dummy = torch.randn(1, WINDOW_LEN, n_feat, device=DEVICE)
    with torch.no_grad():
        for _ in range(20):  # warm-up
            model(dummy)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_trials = 500
        for _ in range(n_trials):
            model(dummy)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    latency_ms = (elapsed / n_trials) * 1000.0
    print(f"  per-window inference latency: {latency_ms:.3f} ms  (batch size 1, {DEVICE.type})")

    return results, latency_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
os.makedirs(DIAG_DIR, exist_ok=True)
orig_dir = os.getcwd()
try:
    os.chdir(DIAG_DIR)
    print(f"Building diagnostic dataset: {len(RUN_TYPES)} run types x {N_RUNS_PER_TYPE} runs "
          f"x {DURATION_S}s @ 1Hz ...")
    t0 = time.time()
    train_df, test_df = build_diagnostic_dataset()
    print(f"  dataset generated in {time.time()-t0:.1f}s "
          f"(train runs={train_df.run_id.nunique()}, test runs={test_df.run_id.nunique()})")
    print("  run_type counts (train):", train_df.groupby("run_id").run_type.first().value_counts().to_dict())
    print("  run_type counts (test): ", test_df.groupby("run_id").run_type.first().value_counts().to_dict())

    print("\nTraining LSTM on combined diagnostic dataset...")
    t0 = time.time()
    results, latency_ms = train_lstm_on(train_df, test_df)
    print(f"LSTM done in {time.time()-t0:.1f}s")

    print("\n=== Combined-attack diagnostic results ===")
    rows = []
    for slice_name, r in results.items():
        if r is None:
            print(f"  {slice_name}: insufficient positive/negative windows, skipped")
            continue
        print(f"  {slice_name:15s}  AUC={r['auc']:.4f}  PR-AUC={r['ap']:.4f}  F1={r['f1']:.4f}  "
              f"n={r['n']}  pos={r['pos']}")
        rows.append(dict(slice=slice_name, auc=r["auc"], pr_auc=r["ap"], f1=r["f1"], n=r["n"], pos=r["pos"]))
    rows.append(dict(slice="inference_latency_ms_batch1", auc=latency_ms, pr_auc=None, f1=None, n=None, pos=None))

    summary = pd.DataFrame(rows)
finally:
    os.chdir(orig_dir)

summary.to_csv("combined_attack_diagnostic_results.csv", index=False)
print("\nSaved combined_attack_diagnostic_results.csv")
shutil.rmtree(os.path.join(orig_dir, DIAG_DIR), ignore_errors=True)
