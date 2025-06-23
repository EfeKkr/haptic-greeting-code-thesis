# Splits .npy feature files into train/val/test sets based on dyad ID to avoid identity overlap.
# Dyads are first extracted and shuffled with a fixed seed for reproducibility.
# The dataset is split using configurable ratios (default: 70% train, 15% val, 15% test).

import os
import random

def split_dyads(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)
    files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    dyads = list(set(f.split('-')[0] for f in files))
    dyads.sort()
    random.shuffle(dyads)

    n = len(dyads)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_dyads = set(dyads[:n_train])
    val_dyads = set(dyads[n_train:n_train+n_val])
    test_dyads = set(dyads[n_train+n_val:])

    def filter_by_dyads(dyad_set):
        return [f for f in files if f.split('-')[0] in dyad_set]

    return (
        filter_by_dyads(train_dyads),
        filter_by_dyads(val_dyads),
        filter_by_dyads(test_dyads)
    )
