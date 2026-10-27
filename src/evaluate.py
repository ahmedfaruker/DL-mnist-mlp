import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dataset import get_dataloaders
from model import MLP

RESULTS_DIR = "results"


def load_model(run_name, device):
    # Rebuild the exact architecture from the saved config, then load best weights
    run_dir = os.path.join(RESULTS_DIR, run_name)
    with open(os.path.join(run_dir, "summary.json")) as f:
        cfg = json.load(f)["config"]
    model = MLP(hidden_sizes=tuple(cfg["hidden"]), dropout=cfg["dropout"])
    state = torch.load(os.path.join(run_dir, "best_model.pt"), map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()  # eval() disables dropout


@torch.no_grad()
def predict(model, loader, device):
    logits, ys, xs = [], [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu())
        ys.append(y)
        xs.append(x)
    return torch.cat(logits), torch.cat(ys), torch.cat(xs)


def plot_confusion(cm, out_path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    for i in range(10):
        for j in range(10):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color=color)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_errors(xs, ys, preds, confs, out_path, title, n=24):
    wrong = (preds != ys).nonzero().flatten()[:n]
    cols = 8
    rows = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 2.2))
    for ax in axes.flatten():
        ax.axis("off")
    for ax, idx in zip(axes.flatten(), wrong):
        ax.imshow(xs[idx].squeeze(), cmap="gray")
        ax.set_title(f"T:{ys[idx]} P:{preds[idx]}\n{confs[idx]*100:.0f}%", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["exp1_baseline", "exp10_final"])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, _ = get_dataloaders()
    test_loader = loaders["test"]

    for run in args.runs:
        run_dir = os.path.join(RESULTS_DIR, run)
        model = load_model(run, device)
        logits, ys, xs = predict(model, test_loader, device)

        probs = F.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)
        test_loss = F.cross_entropy(logits, ys).item()
        test_acc = (preds == ys).float().mean().item()

        cm = np.zeros((10, 10), dtype=int)
        np.add.at(cm, (ys.numpy(), preds.numpy()), 1)
        per_class_acc = {str(c): cm[c, c] / cm[c].sum() for c in range(10)}

        result = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "num_errors": int((preds != ys).sum()),
            "per_class_acc": per_class_acc,
            "confusion_matrix": cm.tolist(),
        }
        with open(os.path.join(run_dir, "test_results.json"), "w") as f:
            json.dump(result, f, indent=2)

        plot_confusion(cm, os.path.join(run_dir, "confusion_matrix.png"), f"{run} - test")
        plot_errors(xs, ys, preds, confs, os.path.join(run_dir, "test_errors.png"), f"{run} - misclassified")

        worst = sorted(per_class_acc.items(), key=lambda kv: kv[1])[:3]
        print(f"{run:15s} | test loss {test_loss:.4f} | test acc {test_acc*100:.2f}% | "
              f"errors {result['num_errors']}/10000 | worst classes "
              + ", ".join(f"{c}:{a*100:.1f}%" for c, a in worst))


if __name__ == "__main__":
    main()