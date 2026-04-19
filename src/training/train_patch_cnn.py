from __future__ import annotations

import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================
EXPERIMENT_NAME = "full_dataset_multispectral_+_VIs_2"

DATASET_INDEX = Path(
    r"C:\Users\franc\Desktop\thesis_project\processed_patches_2\full_dataset\dataset_index_full.csv"
)

OUTPUT_DIR = Path(r"C:\Users\franc\Desktop\thesis_project\cnn_runs") / EXPERIMENT_NAME

SEED = 42
BATCH_SIZE = 16
NUM_EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4

VAL_RATIO = 0.2
TEST_RATIO = 0.2

NUM_WORKERS = 0   # keep 0 on Windows first
PIN_MEMORY = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_CLASS_WEIGHTS = True

# Current export order:
# RGB(3) + MS5(5) + VI5(5) + MASK(1) = 14 channels
# VI order: GNDVI, GCI, NDREI, NRI, GI
USE_CHANNELS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


# ============================================================
# UTILS
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def build_group_key(df: pd.DataFrame) -> pd.Series:
    """
    Use field_id + plant_id as unique group key so plants from different
    fields/dates with the same plant_id do not leak across splits.
    """
    if "field_id" not in df.columns:
        raise RuntimeError(
            "dataset_index is missing 'field_id'. "
            "Rebuild patches with the multi-field build_patch_dataset.py."
        )
    return df["field_id"].astype(str) + "__" + df["plant_id"].astype(str)


def split_by_group(df: pd.DataFrame, val_ratio=0.2, test_ratio=0.2, seed=42):
    rng = random.Random(seed)

    group_keys = sorted(build_group_key(df).unique().tolist())
    rng.shuffle(group_keys)

    n = len(group_keys)
    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))

    test_groups = set(group_keys[:n_test])
    val_groups = set(group_keys[n_test:n_test + n_val])
    train_groups = set(group_keys[n_test + n_val:])

    df = df.copy()
    df["group_key"] = build_group_key(df)

    train_df = df[df["group_key"].isin(train_groups)].reset_index(drop=True)
    val_df = df[df["group_key"].isin(val_groups)].reset_index(drop=True)
    test_df = df[df["group_key"].isin(test_groups)].reset_index(drop=True)

    return train_df, val_df, test_df


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def compute_channel_stats(df: pd.DataFrame, use_channels=None, mask_channel_idx_in_selected=None):
    """
    Compute mean/std on TRAIN ONLY, excluding the mask channel if provided.
    Returns:
        mean: (C,)
        std:  (C,)
    """
    n = 0
    channel_sum = None
    channel_sq_sum = None
    C = None

    for _, row in df.iterrows():
        data = np.load(row["npz_path"], allow_pickle=False)
        x = data["x"].astype(np.float32)   # HWC

        if use_channels is not None:
            x = x[:, :, use_channels]

        flat = x.reshape(-1, x.shape[-1])  # (pixels, C)

        if C is None:
            C = flat.shape[1]

        if mask_channel_idx_in_selected is not None:
            keep = [i for i in range(flat.shape[1]) if i != mask_channel_idx_in_selected]
            flat_stats = flat[:, keep]
        else:
            flat_stats = flat

        if channel_sum is None:
            channel_sum = flat_stats.sum(axis=0)
            channel_sq_sum = (flat_stats ** 2).sum(axis=0)
        else:
            channel_sum += flat_stats.sum(axis=0)
            channel_sq_sum += (flat_stats ** 2).sum(axis=0)

        n += flat_stats.shape[0]

    mean_cont = channel_sum / n
    var_cont = channel_sq_sum / n - mean_cont ** 2
    std_cont = np.sqrt(np.maximum(var_cont, 1e-8))

    mean = np.zeros(C, dtype=np.float32)
    std = np.ones(C, dtype=np.float32)

    if mask_channel_idx_in_selected is None:
        mean[:] = mean_cont.astype(np.float32)
        std[:] = std_cont.astype(np.float32)
    else:
        keep = [i for i in range(C) if i != mask_channel_idx_in_selected]
        mean[keep] = mean_cont.astype(np.float32)
        std[keep] = std_cont.astype(np.float32)
        mean[mask_channel_idx_in_selected] = 0.0
        std[mask_channel_idx_in_selected] = 1.0

    return mean, std


# ============================================================
# DATASET
# ============================================================

class PatchDataset(Dataset):
    def __init__(self, df: pd.DataFrame, use_channels=None, augment=False, channel_mean=None, channel_std=None):
        self.df = df.reset_index(drop=True)
        self.use_channels = use_channels
        self.augment = augment
        self.channel_mean = np.array(channel_mean, dtype=np.float32).reshape(1, 1, -1)
        self.channel_std = np.array(channel_std, dtype=np.float32).reshape(1, 1, -1)

    def __len__(self):
        return len(self.df)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            x = np.flip(x, axis=1).copy()
        if random.random() < 0.5:
            x = np.flip(x, axis=0).copy()
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])
            x = np.rot90(x, k=k, axes=(0, 1)).copy()
        return x

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npz_path = row["npz_path"]

        data = np.load(npz_path, allow_pickle=False)
        x = data["x"].astype(np.float32)   # HWC
        y = int(data["y"])

        if self.use_channels is not None:
            x = x[:, :, self.use_channels]

        if self.augment:
            x = self._augment(x)

        x = (x - self.channel_mean) / (self.channel_std + 1e-6)

        x = np.transpose(x, (2, 0, 1)).copy()

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# ============================================================
# MODEL
# ============================================================

class SimplePatchCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# TRAIN / EVAL
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = epoch_loss
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = epoch_loss
    return metrics


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)
    ensure_dir(OUTPUT_DIR)

    df = pd.read_csv(DATASET_INDEX)

    print("Using device:", DEVICE)
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if len(df) == 0:
        raise RuntimeError("dataset_index.csv is empty.")

    required_cols = {"field_id", "field_name", "plant_id", "npz_path", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"dataset_index is missing required columns: {sorted(missing)}")

    train_df, val_df, test_df = split_by_group(
        df,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED
    )

    print(f"Total patches: {len(df)}")
    print(f"Train patches: {len(train_df)}")
    print(f"Val patches:   {len(val_df)}")
    print(f"Test patches:  {len(test_df)}")

    print(f"Train unique groups: {train_df['group_key'].nunique()}")
    print(f"Val unique groups:   {val_df['group_key'].nunique()}")
    print(f"Test unique groups:  {test_df['group_key'].nunique()}")

    print(f"Train fields: {train_df['field_id'].nunique()}")
    print(f"Val fields:   {val_df['field_id'].nunique()}")
    print(f"Test fields:  {test_df['field_id'].nunique()}")

    print("\nClass counts:")
    print("Train:", Counter(train_df["label"].tolist()))
    print("Val:  ", Counter(val_df["label"].tolist()))
    print("Test: ", Counter(test_df["label"].tolist()))

    first = np.load(df.iloc[0]["npz_path"], allow_pickle=False)
    in_channels = first["x"].shape[-1]
    if USE_CHANNELS is not None:
        in_channels = len(USE_CHANNELS)

    ORIGINAL_MASK_CHANNEL = 13

    mask_channel_idx_in_selected = None
    if USE_CHANNELS is not None and ORIGINAL_MASK_CHANNEL in USE_CHANNELS:
        mask_channel_idx_in_selected = USE_CHANNELS.index(ORIGINAL_MASK_CHANNEL)
    elif USE_CHANNELS is None:
        mask_channel_idx_in_selected = ORIGINAL_MASK_CHANNEL

    channel_mean, channel_std = compute_channel_stats(
        train_df,
        use_channels=USE_CHANNELS,
        mask_channel_idx_in_selected=mask_channel_idx_in_selected
    )

    print("Channel mean:", channel_mean)
    print("Channel std: ", channel_std)
    print("Mask channel in selected:", mask_channel_idx_in_selected)

    train_ds = PatchDataset(
        train_df,
        use_channels=USE_CHANNELS,
        augment=True,
        channel_mean=channel_mean,
        channel_std=channel_std
    )
    val_ds = PatchDataset(
        val_df,
        use_channels=USE_CHANNELS,
        augment=False,
        channel_mean=channel_mean,
        channel_std=channel_std
    )
    test_ds = PatchDataset(
        test_df,
        use_channels=USE_CHANNELS,
        augment=False,
        channel_mean=channel_mean,
        channel_std=channel_std
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    model = SimplePatchCNN(in_channels=in_channels, num_classes=2).to(DEVICE)

    if USE_CLASS_WEIGHTS:
        train_counts = Counter(train_df["label"].tolist())
        n0 = train_counts.get(0, 1)
        n1 = train_counts.get(1, 1)
        class_weights = torch.tensor(
            [1.0 / n0, 1.0 / n1],
            dtype=torch.float32,
            device=DEVICE
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using class weights: {class_weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_f1 = -1.0
    best_path = OUTPUT_DIR / "best_model.pt"
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items() if k != "confusion_matrix"},
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"},
        }
        history.append(row)

        print(
            f"\nEpoch {epoch}/{NUM_EPOCHS}"
            f"\n  train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} "
            f"prec={train_metrics['precision']:.4f} rec={train_metrics['recall']:.4f} f1={train_metrics['f1']:.4f}"
            f"\n  val   loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
            f"prec={val_metrics['precision']:.4f} rec={val_metrics['recall']:.4f} f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_path)
            print(f"  -> saved best model to {best_path}")

    pd.DataFrame(history).to_csv(OUTPUT_DIR / "history.csv", index=False)

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    test_metrics = evaluate(model, test_loader, criterion, DEVICE)

    print("\n=== FINAL TEST ===")
    print(json.dumps(test_metrics, indent=2))

    with open(OUTPUT_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    split_info = {
        "train_groups": sorted(train_df["group_key"].unique().tolist()),
        "val_groups": sorted(val_df["group_key"].unique().tolist()),
        "test_groups": sorted(test_df["group_key"].unique().tolist()),
        "train_fields": sorted(train_df["field_id"].unique().tolist()),
        "val_fields": sorted(val_df["field_id"].unique().tolist()),
        "test_fields": sorted(test_df["field_id"].unique().tolist()),
    }
    with open(OUTPUT_DIR / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nDone. Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()