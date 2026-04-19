from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import rasterio


# ============================================================
# CONFIG
# ============================================================

PLANT_DIR = Path(r"C:\Users\franc\Desktop\thesis_project\data_aligned\Carrù 22-6-22\Pianta 2")

CENTER_CROP = 1200


# ============================================================
# FILE DISCOVERY
# ============================================================

def find_plant_files(plant_dir: Path):
    rgb_candidates = (
        list(plant_dir.glob("DJI_*0.JPG")) +
        list(plant_dir.glob("DJI_*0.jpg")) +
        list(plant_dir.glob("DJI_*0.JPEG")) +
        list(plant_dir.glob("DJI_*0.jpeg"))
    )
    if not rgb_candidates:
        raise FileNotFoundError(f"No RGB file found in {plant_dir}")

    rgb_path = rgb_candidates[0]
    stem = rgb_path.stem
    prefix = stem[:-1]

    def tif_for(digit: int) -> Path:
        cands = list(plant_dir.glob(f"{prefix}{digit}.TIF")) + list(plant_dir.glob(f"{prefix}{digit}.tif"))
        if not cands:
            raise FileNotFoundError(f"Missing {prefix}{digit}.TIF in {plant_dir}")
        return cands[0]

    return {
        "rgb": rgb_path,
        "blu": tif_for(1),
        "green": tif_for(2),
        "red": tif_for(3),
        "re": tif_for(4),
        "nir": tif_for(5),
    }


# ============================================================
# IO / NORMALIZATION
# ============================================================

def load_rgb(path: Path):
    # Unicode-safe image loading for Windows paths like "Carrù ..."
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to load RGB image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_band(path: Path):
    with rasterio.open(path) as src:
        return src.read(1)


def normalize01(img, p_lo=2, p_hi=98):
    img = img.astype(np.float32)
    lo = np.percentile(img, p_lo)
    hi = np.percentile(img, p_hi)

    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)

    out = (img - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


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


# ============================================================
# VIS HELPERS
# ============================================================

def make_overlay(reference, moving):
    ref = normalize01(reference)
    mov = normalize01(moving)

    overlay = np.zeros((ref.shape[0], ref.shape[1], 3), dtype=np.float32)
    overlay[:, :, 0] = ref
    overlay[:, :, 1] = mov
    return overlay


def abs_diff(a, b):
    a = normalize01(a)
    b = normalize01(b)
    return np.abs(a - b)


def show_rgb(rgb):
    plt.figure(figsize=(7, 7))
    plt.imshow(center_crop(rgb, CENTER_CROP))
    plt.title("RGB reference")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_band_vs_nir(nir, band, band_name):
    nir_c = center_crop(nir, CENTER_CROP)
    band_c = center_crop(band, CENTER_CROP)
    overlay_c = center_crop(make_overlay(nir, band), CENTER_CROP)
    diff_c = center_crop(abs_diff(nir, band), CENTER_CROP)

    fig, ax = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(f"{band_name} vs NIR", fontsize=14)

    ax[0].imshow(nir_c, cmap="gray")
    ax[0].set_title("NIR")
    ax[0].axis("off")

    ax[1].imshow(band_c, cmap="gray")
    ax[1].set_title(band_name)
    ax[1].axis("off")

    ax[2].imshow(overlay_c)
    ax[2].set_title("Overlay\n(red=NIR, green=band)")
    ax[2].axis("off")

    ax[3].imshow(diff_c, cmap="magma")
    ax[3].set_title("Abs diff")
    ax[3].axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Verifying aligned plant folder:\n{PLANT_DIR}\n")

    files = find_plant_files(PLANT_DIR)

    for k, v in files.items():
        print(f"{k}: {v}")

    rgb = load_rgb(files["rgb"])
    nir = load_band(files["nir"])
    blu = load_band(files["blu"])
    green = load_band(files["green"])
    red = load_band(files["red"])
    re = load_band(files["re"])

    print("\nShapes:")
    print("RGB  :", rgb.shape)
    print("BLU  :", blu.shape, blu.dtype)
    print("GREEN:", green.shape, green.dtype)
    print("RED  :", red.shape, red.dtype)
    print("RE   :", re.shape, re.dtype)
    print("NIR  :", nir.shape, nir.dtype)

    show_rgb(rgb)

    show_band_vs_nir(nir, blu, "BLU")
    show_band_vs_nir(nir, green, "GREEN")
    show_band_vs_nir(nir, red, "RED")
    show_band_vs_nir(nir, re, "RE")


if __name__ == "__main__":
    main()