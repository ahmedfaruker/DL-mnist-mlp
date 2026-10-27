import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")  # save plots to file, no GUI window
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import MLP, count_params

RESULTS_DIR = "results"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="exp1_baseline")
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)  # only used by SGD
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    # Training pass if an optimizer is given, otherwise evaluation pass (no gradients)
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()   # clear old gradients
                loss.backward()         # compute new gradients
                optimizer.step()        # update weights
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def plot_history(history, best_epoch, out_path, title):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_title("Loss")
    ax1.set_ylabel("cross-entropy")

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="train")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="val")
    ax2.set_title("Accuracy")
    ax2.set_ylabel("%")

    for ax in (ax1, ax2):
        ax.axvline(best_epoch, color="gray", linestyle="--", label=f"best val (ep {best_epoch})")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(RESULTS_DIR, args.name)
    os.makedirs(run_dir, exist_ok=True)

    loaders, _ = get_dataloaders(batch_size=args.batch_size)  # test loader is NOT used here
    model = MLP(hidden_sizes=tuple(args.hidden), dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    print(f"device={device} | params={count_params(model)} | config={vars(args)}")

    history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "epoch_time"]}
    best_val_acc, best_epoch = 0.0, -1
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, loaders["val"], criterion, device)
        ep_time = time.perf_counter() - t0

        for k, v in zip(history.keys(), [tr_loss, va_loss, tr_acc, va_acc, ep_time]):
            history[k].append(v)

        # Model selection is based on validation accuracy only (never test)
        if va_acc > best_val_acc:
            best_val_acc, best_epoch = va_acc, epoch
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))

        print(f"ep {epoch:2d} | train loss {tr_loss:.4f} acc {tr_acc*100:.2f}% | "
              f"val loss {va_loss:.4f} acc {va_acc*100:.2f}% | {ep_time:.1f}s")

    total_time = time.perf_counter() - start

    summary = {
        "config": vars(args),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "params": count_params(model),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "total_train_time_sec": total_time,
        "history": history,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plot_history(history, best_epoch, os.path.join(run_dir, "curves.png"), args.name)
    print(f"\nbest val acc {best_val_acc*100:.2f}% at epoch {best_epoch} | total time {total_time:.1f}s")
    print(f"saved to {run_dir}/")


if __name__ == "__main__":
    main()