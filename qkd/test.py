from generate_qkd_dataset import run_pulses_per_tick_ablation
summary = run_pulses_per_tick_ablation()
summary.to_csv("pulses_per_tick_ablation.csv", index=False)