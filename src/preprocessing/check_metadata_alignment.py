from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

import preprocess_segmentation_v2 as segv2


FIELD_DIR = Path(r"C:\Users\franc\Desktop\thesis_project\data\Carrù 22-6-22")
PLANT_DIR = FIELD_DIR / "Pianta 2"


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
    stem = rgb_path.stem  # e.g. DJI_0010
    prefix = stem[:-1]    # e.g. DJI_001

    def tif_for(digit: int) -> Path:
        cands = list(plant_dir.glob(f"{prefix}{digit}.TIF")) + list(plant_dir.glob(f"{prefix}{digit}.tif"))
        if not cands:
            raise FileNotFoundError(f"Missing {prefix}{digit}.TIF")
        return cands[0]

    return {
        "blu": tif_for(1),
        "green": tif_for(2),
        "red": tif_for(3),
        "re": tif_for(4),
        "nir": tif_for(5),
    }


def read_dji_offsets(tif_path: Path):
    cmd = [
        "exiftool",
        "-j",
        "-BandName",
        "-RelativeOpticalCenterX",
        "-RelativeOpticalCenterY",
        str(tif_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    meta = json.loads(result.stdout)[0]
    return {
        "band_name": meta.get("BandName"),
        "rel_x": float(meta.get("RelativeOpticalCenterX", 0.0)),
        "rel_y": float(meta.get("RelativeOpticalCenterY", 0.0)),
    }


def normalize01(img: np.ndarray, p_lo=2, p_hi=98):
    lo = np.percentile(img, p_lo)
    hi = np.percentile(img, p_hi)
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    out = (img.astype(np.float32) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def shift_image(img: np.ndarray, dx: float, dy: float):
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(
        img.astype(np.float32),
        M,
        (img.shape[1], img.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def false_overlay(ref01: np.ndarray, mov01: np.ndarray):
    z = np.zeros_like(ref01)
    return np.dstack([ref01, mov01, z])


def canny(img01: np.ndarray):
    u8 = (img01 * 255).astype(np.uint8)
    return cv2.Canny(u8, 60, 140)


def edge_overlay(base01: np.ndarray, e1: np.ndarray, e2: np.ndarray):
    out = np.dstack([base01, base01, base01]).copy()
    m1 = e1 > 0
    m2 = e2 > 0
    out[m1, 0] = 1.0
    out[m1, 1] *= 0.2
    out[m1, 2] *= 0.2
    out[m2, 1] = 1.0
    out[m2, 0] = np.maximum(out[m2, 0], 0.2)
    out[m2, 2] *= 0.2
    return np.clip(out, 0.0, 1.0)

def ecc_refine_translation(moving: np.ndarray, reference: np.ndarray, n_iter: int = 100, eps: float = 1e-6):
    """
    Refine alignment of `moving` to `reference` using ECC with translation model.
    Both images should be float32 in [0,1].
    Returns:
        aligned, warp_matrix, cc
    """
    ref32 = reference.astype(np.float32)
    mov32 = moving.astype(np.float32)

    # optional mild blur helps ECC stability
    ref_blur = cv2.GaussianBlur(ref32, (5, 5), 0)
    mov_blur = cv2.GaussianBlur(mov32, (5, 5), 0)

    warp = np.eye(2, 3, dtype=np.float32)  # translation model
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        n_iter,
        eps
    )

    try:
        cc, warp = cv2.findTransformECC(
            templateImage=ref_blur,
            inputImage=mov_blur,
            warpMatrix=warp,
            motionType=cv2.MOTION_TRANSLATION,
            criteria=criteria,
            inputMask=None,
            gaussFiltSize=5
        )

        aligned = cv2.warpAffine(
            mov32,
            warp,
            (reference.shape[1], reference.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        return aligned, warp, cc

    except cv2.error as e:
        print(f"[WARN] ECC failed: {e}")
        return mov32, np.eye(2, 3, dtype=np.float32), None


def show_alignment(ref, mov, meta, title_prefix):
    ref01 = normalize01(ref)
    mov01 = normalize01(mov)

    # Step 1: metadata shift
    mov_meta = shift_image(mov01, -meta["rel_x"], -meta["rel_y"])

    # Step 2: ECC refinement on top of metadata shift
    mov_meta_ecc, warp_ecc, cc = ecc_refine_translation(mov_meta, ref01)

    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(
        f'{title_prefix} | band={meta["band_name"]} | '
        f'rel_x={meta["rel_x"]:.4f}, rel_y={meta["rel_y"]:.4f} | '
        f'ECC cc={cc if cc is not None else "FAIL"} | '
        f'warp=({warp_ecc[0,2]:.3f}, {warp_ecc[1,2]:.3f})',
        fontsize=13
    )

    # Overlays
    ax[0, 0].imshow(false_overlay(ref01, mov01))
    ax[0, 0].set_title("Before")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(false_overlay(ref01, mov_meta))
    ax[0, 1].set_title("After metadata")
    ax[0, 1].axis("off")

    ax[0, 2].imshow(false_overlay(ref01, mov_meta_ecc))
    ax[0, 2].set_title("After metadata + ECC")
    ax[0, 2].axis("off")

    ax[0, 3].imshow(ref01, cmap="gray")
    ax[0, 3].set_title("Reference (NIR)")
    ax[0, 3].axis("off")

    # Edge overlays
    e_ref = canny(ref01)
    e_before = canny(mov01)
    e_meta = canny(mov_meta)
    e_meta_ecc = canny(mov_meta_ecc)

    ax[1, 0].imshow(edge_overlay(ref01, e_ref, e_before))
    ax[1, 0].set_title("Edges before")
    ax[1, 0].axis("off")

    ax[1, 1].imshow(edge_overlay(ref01, e_ref, e_meta))
    ax[1, 1].set_title("Edges after metadata")
    ax[1, 1].axis("off")

    ax[1, 2].imshow(edge_overlay(ref01, e_ref, e_meta_ecc))
    ax[1, 2].set_title("Edges after metadata + ECC")
    ax[1, 2].axis("off")

    ax[1, 3].imshow(mov_meta_ecc, cmap="gray")
    ax[1, 3].set_title("Aligned moving band")
    ax[1, 3].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    files = find_plant_files(PLANT_DIR)

    red = segv2.read_singleband_tif(files["red"])
    nir = segv2.read_singleband_tif(files["nir"])
    re_ = segv2.read_singleband_tif(files["re"])

    red_meta = read_dji_offsets(files["red"])
    re_meta = read_dji_offsets(files["re"])

    print("RED meta:", red_meta)
    print("RE  meta:", re_meta)

    show_alignment(nir, red, red_meta, "NIR vs RED")
    show_alignment(nir, re_, re_meta, "NIR vs RedEdge")


if __name__ == "__main__":
    main()