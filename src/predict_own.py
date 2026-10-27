import argparse
import glob
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

from evaluate import load_model

PROCESSED_DIR = "my_digits/processed"
OUT_DIR = "my_digits"
RESULTS_DIR = "results"


def load_own_digits():
    paths = sorted(glob.glob(os.path.join(PROCESSED_DIR, "digit_*.png")))
    return np.stack([cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths])  # [N, 28, 28] uint8


def normalize(imgs):
    # Same pipeline as training: [0,255] -> [0,1] -> (x - mean) / std using TRAIN-split stats
    with open(os.path.join(RESULTS_DIR, "norm_stats.json")) as f:
        s = json.load(f)
    x = torch.from_numpy(imgs).float().div(255.0).unsqueeze(1)  # [N, 1, 28, 28]
    return (x - s["mean"]) / s["std"]


@torch.no_grad()
def predict(run, x, device):
    model = load_model(run, device)
    return F.softmax(model(x.to(device)), dim=1).cpu()


def plot_predictions(imgs, truth, all_probs, out_path):
    runs = list(all_probs)
    n = len(imgs)
    fig, axes = plt.subplots(len(runs), n, figsize=(n * 1.4, len(runs) * 2.1), squeeze=False)
    for r, run in enumerate(runs):
        top = all_probs[run].topk(2, dim=1)
        for i in range(n):
            ax = axes[r, i]
            ax.imshow(imgs[i], cmap="gray")
            ax.axis("off")
            p1, p2 = top.indices[i].tolist()
            c1, c2 = top.values[i].tolist()
            color = "green" if p1 == truth[i] else "red"
            ax.set_title(f"P:{p1} {c1:.0%}\n({p2} {c2:.0%})", fontsize=8, color=color)
        axes[r, 0].text(-0.25, 0.5, run, transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=9)
    fig.suptitle("Own handwriting predictions (green = correct, red = wrong; 2nd guess in parentheses)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_vs_mnist(imgs, truth, out_path, k=2, seed=0):
    # Visual distribution-shift check: own digit vs random MNIST TRAIN samples of the same class
    mnist = datasets.MNIST("data", train=True, download=True)
    rng = np.random.default_rng(seed)
    n = len(imgs)
    fig, axes = plt.subplots(k + 1, n, figsize=(n * 1.3, (k + 1) * 1.5), squeeze=False)
    for i, t in enumerate(truth):
        axes[0, i].imshow(imgs[i], cmap="gray")
        axes[0, i].set_title(f"own {t}", fontsize=8)
        candidates = (mnist.targets == t).nonzero().flatten().numpy()
        for j, idx in enumerate(rng.choice(candidates, size=k, replace=False)):
            axes[j + 1, i].imshow(mnist.data[idx].numpy(), cmap="gray")
            axes[j + 1, i].set_title("MNIST", fontsize=8)
    for ax in axes.flatten():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Simple numeric shift indicator: average "ink" per image
    own_ink = imgs.mean() / 255.0
    mnist_ink = mnist.data.float().mean().item() / 255.0
    print(f"\nmean pixel intensity (ink): own {own_ink:.4f} | MNIST train {mnist_ink:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--truth", default="23120205041")
    p.add_argument("--runs", nargs="+", default=["exp10_final", "exp1_baseline"])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imgs = load_own_digits()
    truth = [int(c) for c in args.truth]
    assert len(truth) == len(imgs), f"found {len(imgs)} digits but truth has {len(truth)}"
    x = normalize(imgs)

    all_probs, results = {}, {}
    for run in args.runs:
        probs = predict(run, x, device)
        all_probs[run] = probs
        top = probs.topk(2, dim=1)
        preds = top.indices[:, 0].tolist()
        n_ok = sum(p == t for p, t in zip(preds, truth))
        pred_str = "".join(map(str, preds))
        print(f"\n{run}: predicted {pred_str} | true {args.truth} | "
              f"{n_ok}/{len(truth)} correct ({n_ok / len(truth) * 100:.1f}%)")
        for i, t in enumerate(truth):
            (p1, p2), (c1, c2) = top.indices[i].tolist(), top.values[i].tolist()
            print(f"  [{i:2d}] true {t} -> pred {p1} ({c1:6.1%}) | 2nd {p2} ({c2:6.1%})"
                  f"  {'OK' if p1 == t else 'WRONG'}")
        results[run] = {
            "predicted": pred_str,
            "true": args.truth,
            "accuracy": n_ok / len(truth),
            "per_digit": [
                {"true": t, "pred": top.indices[i, 0].item(), "conf": top.values[i, 0].item(),
                 "second": top.indices[i, 1].item(), "second_conf": top.values[i, 1].item()}
                for i, t in enumerate(truth)
            ],
        }

    with open(os.path.join(OUT_DIR, "predictions.json"), "w") as f:
        json.dump(results, f, indent=2)
    plot_predictions(imgs, truth, all_probs, os.path.join(OUT_DIR, "predictions.png"))
    plot_vs_mnist(imgs, truth, os.path.join(OUT_DIR, "own_vs_mnist.png"))
    print(f"\nsaved {OUT_DIR}/predictions.json, predictions.png, own_vs_mnist.png")


if __name__ == "__main__":
    main()