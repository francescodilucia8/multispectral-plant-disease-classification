import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio


# ============================================================
# CONFIG
# ============================================================

DATASET_INDEX = r"C:\Users\franc\Desktop\thesis_project\processed_patches\carru_2206\dataset_index.csv"

RAW_DATA_ROOT = Path(r"C:\Users\franc\Desktop\thesis_project\data")
ALIGNED_DATA_ROOT = Path(r"C:\Users\franc\Desktop\thesis_project\data_aligned")

RANDOM_MODE = True
DISPLAY_CENTER_CROP = 1200
REFERENCE_BAND = "NIR"
BANDS_TO_CHECK = ["GREEN", "RED", "REDEDGE"]


# ============================================================
# BASIC HELPERS
# ============================================================

def percentile_normalize(img, p_low=1, p_high=99):
    img = img.astype(np.float32)
    lo = np.percentile(img, p_low)
    hi = np.percentile(img, p_high)
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    out = (img - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    return out.astype(np.float32)


def center_crop(img, crop_size):
    if crop_size is None:
        return img

    h, w = img.shape[:2]
    ch = min(crop_size, h)
    cw = min(crop_size, w)

    y0 = max(0, h // 2 - ch // 2)
    x0 = max(0, w // 2 - cw // 2)

    if img.ndim == 2:
        return img[y0:y0 + ch, x0:x0 + cw]
    return img[y0:y0 + ch, x0:x0 + cw, :]


def resize_to_shape(img, shape_hw):
    h, w = shape_hw
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def load_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
    return arr


def load_rgb(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to load RGB image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def make_overlay(reference, moving):
    ref = percentile_normalize(reference)
    mov = percentile_normalize(moving)

    overlay = np.zeros((ref.shape[0], ref.shape[1], 3), dtype=np.float32)
    overlay[:, :, 0] = ref
    overlay[:, :, 1] = mov
    return overlay


def abs_diff(a, b):
    a = percentile_normalize(a)
    b = percentile_normalize(b)
    return np.abs(a - b)


# ============================================================
# META / PATH DISCOVERY
# ============================================================

def infer_sample_dir_from_npz(npz_path):
    p = Path(npz_path)
    if p.parent.name.lower() == "patches":
        return p.parent.parent
    return p.parent


def load_meta_for_row(row):
    if "meta_json_path" in row.index and pd.notna(row["meta_json_path"]):
        meta_path = Path(row["meta_json_path"])
    else:
        p = Path(row["npz_path"])
        candidates = [
            p.parent / "meta.json",
            p.parent.parent / "meta.json",
        ]

        meta_path = None
        for cand in candidates:
            if cand.exists():
                meta_path = cand
                break

        if meta_path is None:
            raise FileNotFoundError(
                "meta.json not found. Tried:\n" + "\n".join(str(c) for c in candidates)
            )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta, meta_path


def discover_aligned_paths_from_meta(meta, meta_path):
    rgb_path = None
    band_paths = {
        "GREEN": None,
        "RED": None,
        "REDEDGE": None,
        "NIR": None,
    }

    if "files" in meta and isinstance(meta["files"], dict):
        files = meta["files"]

        rgb_path = files.get("rgb", None)
        band_paths["GREEN"] = files.get("green", None)
        band_paths["RED"] = files.get("red", None)
        band_paths["REDEDGE"] = files.get("re", None)
        band_paths["NIR"] = files.get("nir", None)

    def resolve_path(p):
        if p is None:
            return None
        p = Path(p)
        if not p.is_absolute():
            p = (meta_path.parent / p).resolve()
        return p

    rgb_path = resolve_path(rgb_path)
    for k in band_paths:
        band_paths[k] = resolve_path(band_paths[k])

    return rgb_path, band_paths


def map_aligned_path_to_raw(aligned_path: Path) -> Path:
    aligned_path = aligned_path.resolve()
    return RAW_DATA_ROOT / aligned_path.relative_to(ALIGNED_DATA_ROOT)


# ============================================================
# PLOTTING
# ============================================================

def plot_one_band_result(ref_name, ref_img, band_name, raw_img, aligned_img,
                         plant_id=None, patch_id=None):
    ref_crop = center_crop(ref_img, DISPLAY_CENTER_CROP)
    raw_crop = center_crop(raw_img, DISPLAY_CENTER_CROP)
    aligned_crop = center_crop(aligned_img, DISPLAY_CENTER_CROP)

    overlay_before = center_crop(make_overlay(ref_img, raw_img), DISPLAY_CENTER_CROP)
    overlay_after = center_crop(make_overlay(ref_img, aligned_img), DISPLAY_CENTER_CROP)

    diff_before = center_crop(abs_diff(ref_img, raw_img), DISPLAY_CENTER_CROP)
    diff_after = center_crop(abs_diff(ref_img, aligned_img), DISPLAY_CENTER_CROP)

    fig, ax = plt.subplots(2, 4, figsize=(18, 9))

    title = f"{band_name} -> {ref_name}"
    if plant_id is not None or patch_id is not None:
        title += f" | plant={plant_id} patch={patch_id}"
    fig.suptitle(title, fontsize=15)

    ax[0, 0].imshow(ref_crop, cmap="gray")
    ax[0, 0].set_title(f"Reference: {ref_name}")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(raw_crop, cmap="gray")
    ax[0, 1].set_title(f"Raw {band_name}")
    ax[0, 1].axis("off")

    ax[0, 2].imshow(overlay_before)
    ax[0, 2].set_title("Overlay BEFORE\n(red=ref, green=raw)")
    ax[0, 2].axis("off")

    ax[0, 3].imshow(diff_before, cmap="magma")
    ax[0, 3].set_title("Abs diff BEFORE")
    ax[0, 3].axis("off")

    ax[1, 0].imshow(ref_crop, cmap="gray")
    ax[1, 0].set_title(f"Reference: {ref_name}")
    ax[1, 0].axis("off")

    ax[1, 1].imshow(aligned_crop, cmap="gray")
    ax[1, 1].set_title(f"Saved aligned {band_name}")
    ax[1, 1].axis("off")

    ax[1, 2].imshow(overlay_after)
    ax[1, 2].set_title("Overlay AFTER\n(red=ref, green=aligned)")
    ax[1, 2].axis("off")

    ax[1, 3].imshow(diff_after, cmap="magma")
    ax[1, 3].set_title("Abs diff AFTER")
    ax[1, 3].axis("off")

    plt.tight_layout()
    plt.show()


def show_rgb_reference(rgb, plant_id=None, patch_id=None):
    rgb_crop = center_crop(rgb, DISPLAY_CENTER_CROP)
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_crop)
    title = "RGB visual reference"
    if plant_id is not None or patch_id is not None:
        title += f" | plant={plant_id} patch={patch_id}"
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN SAMPLE INSPECTION
# ============================================================

def inspect_row(row):
    print("\n" + "#" * 100)
    print("Inspecting row:")
    for col in row.index:
        print(f"  {col}: {row[col]}")

    meta, meta_path = load_meta_for_row(row)
    print(f"\nLoaded meta.json from:\n  {meta_path}")

    aligned_rgb_path, aligned_band_paths = discover_aligned_paths_from_meta(meta, meta_path)

    raw_rgb_path = map_aligned_path_to_raw(aligned_rgb_path) if aligned_rgb_path is not None else None
    raw_band_paths = {
        k: map_aligned_path_to_raw(v) if v is not None else None
        for k, v in aligned_band_paths.items()
    }

    print("\nAligned paths:")
    print(f"  RGB: {aligned_rgb_path}")
    for k, v in aligned_band_paths.items():
        print(f"  {k}: {v}")

    print("\nRaw paths:")
    print(f"  RGB: {raw_rgb_path}")
    for k, v in raw_band_paths.items():
        print(f"  {k}: {v}")

    if aligned_band_paths.get(REFERENCE_BAND) is None:
        raise RuntimeError(f"Reference band path for {REFERENCE_BAND} not found")

    ref_img = load_band(aligned_band_paths[REFERENCE_BAND])
    ref_img = percentile_normalize(ref_img)

    if aligned_rgb_path is not None and Path(aligned_rgb_path).exists():
        try:
            rgb = load_rgb(aligned_rgb_path)
            show_rgb_reference(
                rgb,
                plant_id=row["plant_id"] if "plant_id" in row.index else None,
                patch_id=row["patch_id"] if "patch_id" in row.index else None
            )
        except Exception as e:
            print(f"Could not load RGB reference: {e}")

    for band_name in BANDS_TO_CHECK:
        if band_name == REFERENCE_BAND:
            continue

        raw_band_path = raw_band_paths.get(band_name)
        aligned_band_path = aligned_band_paths.get(band_name)

        if raw_band_path is None or not Path(raw_band_path).exists():
            print(f"\nSkipping {band_name}: raw path not found")
            continue

        if aligned_band_path is None or not Path(aligned_band_path).exists():
            print(f"\nSkipping {band_name}: aligned path not found")
            continue

        raw_img = load_band(raw_band_path)
        aligned_img = load_band(aligned_band_path)

        if raw_img.shape != ref_img.shape:
            raw_img = resize_to_shape(raw_img, ref_img.shape)
        if aligned_img.shape != ref_img.shape:
            aligned_img = resize_to_shape(aligned_img, ref_img.shape)

        raw_img = percentile_normalize(raw_img)
        aligned_img = percentile_normalize(aligned_img)

        plot_one_band_result(
            ref_name=REFERENCE_BAND,
            ref_img=ref_img,
            band_name=band_name,
            raw_img=raw_img,
            aligned_img=aligned_img,
            plant_id=row["plant_id"] if "plant_id" in row.index else None,
            patch_id=row["patch_id"] if "patch_id" in row.index else None,
        )


def main():
    df = pd.read_csv(DATASET_INDEX)

    print("Loaded dataset index:")
    print(f"  {DATASET_INDEX}")
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    idx = 0

    while True:
        if RANDOM_MODE:
            idx = random.randint(0, len(df) - 1)

        row = df.iloc[idx]

        try:
            inspect_row(row)
        except Exception as e:
            print("\nERROR while inspecting row:")
            print(e)

        cmd = input("\n[ENTER]=next | p=previous | r=random | q=quit : ").strip().lower()

        if cmd == "q":
            break
        elif cmd == "p":
            idx = max(0, idx - 1)
            globals()["RANDOM_MODE"] = False
        elif cmd == "r":
            globals()["RANDOM_MODE"] = True
        else:
            if not RANDOM_MODE:
                idx = (idx + 1) % len(df)


if __name__ == "__main__":
    main()