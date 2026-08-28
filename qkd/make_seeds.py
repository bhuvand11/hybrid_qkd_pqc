from generate_qkd_dataset import build_dataset
import shutil

for seed in [1, 2, 3, 4, 5, 6, 7]:   # edit this list if you want more/fewer seeds
    print(f"\n=== Generating seed {seed} ===")
    build_dataset(n_runs=100, duration_s=86_400, seed=seed)
    shutil.move("qkd_train_decoy.csv", f"qkd_train_decoy_seed{seed}.csv")
    shutil.move("qkd_test_decoy.csv",  f"qkd_test_decoy_seed{seed}.csv")
    print(f"Saved qkd_train_decoy_seed{seed}.csv and qkd_test_decoy_seed{seed}.csv")

print("\nAll seeds done.")
