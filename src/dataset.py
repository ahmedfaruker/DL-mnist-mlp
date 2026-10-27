import json
import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATA_DIR = "data"
RESULTS_DIR = "results"
SEED = 42


def get_dataloaders(batch_size=128, val_size=6000, num_workers=0):
    # Load raw train set once to build a reproducible train/val split
    raw_train = datasets.MNIST(DATA_DIR, train=True, download=True)
    n = len(raw_train)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))
    val_idx = perm[:val_size].tolist()
    train_idx = perm[val_size:].tolist()

    # Normalization stats are computed ONLY on the train split (no leakage)
    train_pixels = raw_train.data[train_idx].float() / 255.0
    mean, std = train_pixels.mean().item(), train_pixels.std().item()

    tf = transforms.Compose([
        transforms.ToTensor(),                 # [0,255] -> [0,1]
        transforms.Normalize((mean,), (std,)),
    ])

    full_train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)

    train_set = Subset(full_train, train_idx)
    val_set = Subset(full_train, val_idx)

    # Save stats: own handwriting images must use the exact same normalization
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "norm_stats.json"), "w") as f:
        json.dump({"mean": mean, "std": std}, f, indent=2)

    loaders = {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return loaders, (mean, std)


if __name__ == "__main__":
    loaders, (mean, std) = get_dataloaders()
    for name, loader in loaders.items():
        print(f"{name:5s}: {len(loader.dataset):6d} samples")
    print(f"mean={mean:.4f}, std={std:.4f}")

    x, y = next(iter(loaders["train"]))
    print("batch:", x.shape, y.shape)  # expect [128, 1, 28, 28], [128]