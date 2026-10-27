import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results"


def load_summaries():
    runs = []
    paths = glob.glob(os.path.join(RESULTS_DIR, "*", "summary.json"))
    # Sort by experiment number so exp10 comes after exp9, not after exp1
    exp_num = lambda p: int(m.group(1)) if (m := re.search(r"exp(\d+)", p)) else 999
    for path in sorted(paths, key=exp_num):
        with open(path) as f:
            s = json.load(f)
        s["name"] = os.path.basename(os.path.dirname(path))
        runs.append(s)
    return runs


def make_table(runs):
    header = ("| Experiment | Hidden | Opt | LR | Batch | Dropout | Params | Best ep | "
              "Best val acc | Final train acc | Final val acc | Min val loss | Final val loss | Time (s) |")
    sep = "|" + "---|" * 14
    rows = [header, sep]
    for r in runs:
        c, h = r["config"], r["history"]
        rows.append(
            f"| {r['name']} | {'-'.join(map(str, c['hidden']))} | {c['optimizer']} | {c['lr']} | "
            f"{c['batch_size']} | {c['dropout']} | {r['params']:,} | {r['best_epoch']} | "
            f"{r['best_val_acc']*100:.2f}% | {r['final_train_acc']*100:.2f}% | "
            f"{r['final_val_acc']*100:.2f}% | {min(h['val_loss']):.4f} | "
            f"{r['final_val_loss']:.4f} | {r['total_train_time_sec']:.1f} |"
        )
    return "\n".join(rows)


def plot_all(runs, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    for r in runs:
        h = r["history"]
        ep = range(1, len(h["val_loss"]) + 1)
        style = "-" if r["name"] == "exp1_baseline" else "--"  # highlight baseline
        ax1.plot(ep, h["val_loss"], style, label=r["name"])
        ax2.plot(ep, [a * 100 for a in h["val_acc"]], style, label=r["name"])

    ax1.set_title("Validation loss")
    ax1.set_ylabel("cross-entropy")
    ax2.set_title("Validation accuracy")
    ax2.set_ylabel("%")
    ax2.set_ylim(bottom=94)
    for ax in (ax1, ax2):
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    runs = load_summaries()
    table = make_table(runs)
    print(table)
    with open(os.path.join(RESULTS_DIR, "comparison.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    plot_all(runs, os.path.join(RESULTS_DIR, "comparison_curves.png"))
    print(f"\nsaved {RESULTS_DIR}/comparison.md and {RESULTS_DIR}/comparison_curves.png")