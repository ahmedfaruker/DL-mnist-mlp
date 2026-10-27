import subprocess
import sys

# Each experiment changes ONE thing relative to exp1_baseline
EXPERIMENTS = {
    "exp2_lr_1e-2": ["--lr", "0.01"],
    "exp3_lr_1e-4": ["--lr", "0.0001"],
    "exp4_sgd": ["--optimizer", "sgd", "--lr", "0.01"],
    "exp5_wide": ["--hidden", "512", "256"],
    "exp6_deep": ["--hidden", "512", "256", "128"],
    "exp7_small": ["--hidden", "64", "32"],
    "exp8_dropout": ["--dropout", "0.3"],
    "exp9_batch512": ["--batch-size", "512"],
}

if __name__ == "__main__":
    for name, extra in EXPERIMENTS.items():
        print(f"\n===== {name} =====", flush=True)
        subprocess.run([sys.executable, "src/train.py", "--name", name, *extra], check=True)