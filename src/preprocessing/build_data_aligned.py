from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import rasterio


SRC_ROOT = Path(r"C:\Users\franc\Desktop\thesis_project\data")
DST_ROOT = Path(r"C:\Users\franc\Desktop\thesis_project\data_aligned")

# Set to None to process all field folders inside SRC_ROOT
# Example: ONLY_FIELD = "Carrù 22-6-22"
ONLY_FIELD = None

# ECC params
ECC_N_ITER = 100
ECC_EPS = 1e-6


# ------------------------------------------------------------
# File discovery
# ------------------------------------------------------------

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
    stem = rgb_path.stem   # e.g. DJI_0010
    prefix = stem[:-1]     # e.g. DJI_001

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


# ------------------------------------------------------------
# EXIF metadata offsets
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Raster IO
# ------------------------------------------------------------

def read_singleband_tif_with_profile(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile.copy()
    return arr, profile


def write_singleband_tif(path: Path, arr: np.ndarray, profile: dict):
    out_profile = profile.copy()
    out_profile.update(
        count=1,
        height=arr.shape[0],
        width=arr.shape[1],
        dtype=str(arr.dtype),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(arr, 1)


# ------------------------------------------------------------
# Alignment helpers
# ------------------------------------------------------------

def normalize01(img: np.ndarray, p_lo=2, p_hi=98):
    lo = np.percentile(img, p_lo)
    hi = np.percentile(img, p_hi)
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    out = (img.astype(np.float32) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def shift_image(img: np.ndarray, dx: float, dy: float, out_shape=None, inverse_map=False):
    """
    Apply a pure translation warp.
    """
    if out_shape is None:
        h, w = img.shape[:2]
    else:
        h, w = out_shape[:2]

    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    flags = cv2.INTER_LINEAR
    if inverse_map:
        flags |= cv2.WARP_INVERSE_MAP

    return cv2.warpAffine(
        img.astype(np.float32),
        M,
        (w, h),
        flags=flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def ecc_refine_translation(moving: np.ndarray, reference: np.ndarray, n_iter: int = 100, eps: float = 1e-6):
    """
    Refine alignment of `moving` to `reference` using ECC with translation model.
    Both images should be float32 in [0,1].
    Returns:
        aligned, warp_matrix, cc
    """
    ref32 = reference.astype(np.float32)
    mov32 = moving.astype(np.float32)

    ref_blur = cv2.GaussianBlur(ref32, (5, 5), 0)
    mov_blur = cv2.GaussianBlur(mov32, (5, 5), 0)

    warp = np.eye(2, 3, dtype=np.float32)
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
        print(f"[WARN] ECC failed for current band: {e}")
        return mov32, np.eye(2, 3, dtype=np.float32), None


def cast_like_original(arr_float: np.ndarray, original_dtype):
    """
    Convert float32 warped image back to original dtype.
    """
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        arr_float = np.rint(arr_float)
        arr_float = np.clip(arr_float, info.min, info.max)
        return arr_float.astype(original_dtype)

    if np.issubdtype(original_dtype, np.floating):
        return arr_float.astype(original_dtype)

    return arr_float.astype(original_dtype)


def align_band_to_nir(moving_raw: np.ndarray, nir_raw: np.ndarray, meta: dict):
    """
    Exact intended sequence:
      1) metadata translation
      2) ECC translation refinement on normalized images
      3) apply the same transforms to the original raw moving band
    """
    # Make sure shapes match the NIR canvas
    out_shape = nir_raw.shape

    # Step 1: metadata shift
    dx_meta = -meta["rel_x"]
    dy_meta = -meta["rel_y"]

    moving_meta_raw = shift_image(moving_raw, dx_meta, dy_meta, out_shape=out_shape, inverse_map=False)

    # ECC is computed on normalized metadata-shifted images
    nir01 = normalize01(nir_raw)
    moving_meta01 = normalize01(moving_meta_raw)

    _, warp_ecc, cc = ecc_refine_translation(
        moving=moving_meta01,
        reference=nir01,
        n_iter=ECC_N_ITER,
        eps=ECC_EPS
    )

    # Step 2: apply ECC warp to the metadata-shifted RAW image
    moving_meta_ecc_raw = cv2.warpAffine(
        moving_meta_raw.astype(np.float32),
        warp_ecc,
        (nir_raw.shape[1], nir_raw.shape[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    aligned_raw = cast_like_original(moving_meta_ecc_raw, moving_raw.dtype)

    dbg = {
        "band_name": meta.get("band_name"),
        "metadata_dx": float(dx_meta),
        "metadata_dy": float(dy_meta),
        "ecc_cc": None if cc is None else float(cc),
        "ecc_dx": float(warp_ecc[0, 2]),
        "ecc_dy": float(warp_ecc[1, 2]),
        "ecc_warp": warp_ecc.tolist(),
    }

    return aligned_raw, dbg


# ------------------------------------------------------------
# Plant processing
# ------------------------------------------------------------

def process_plant(plant_dir: Path, dst_plant_dir: Path):
    print(f"\n[PLANT] {plant_dir}")

    files = find_plant_files(plant_dir)

    # Copy RGB as-is
    dst_plant_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(files["rgb"], dst_plant_dir / files["rgb"].name)

    # Read NIR once, keep unchanged
    nir_raw, nir_profile = read_singleband_tif_with_profile(files["nir"])
    write_singleband_tif(dst_plant_dir / files["nir"].name, nir_raw, nir_profile)

    debug = {
        "source_plant_dir": str(plant_dir),
        "files": {k: str(v) for k, v in files.items()},
        "reference_band": "nir",
        "aligned_bands": {}
    }

    for band_key in ["blu", "green", "red", "re"]:
        src_path = files[band_key]
        print(f"  Aligning {band_key.upper()} -> NIR")

        moving_raw, moving_profile = read_singleband_tif_with_profile(src_path)
        meta = read_dji_offsets(src_path)

        # Resize is not done here on purpose: this script assumes same canvas family
        # If shapes differ, warpAffine with out_shape=nir_raw.shape handles final canvas.
        aligned_raw, dbg = align_band_to_nir(moving_raw, nir_raw, meta)

        # Important: use NIR canvas/profile so all aligned bands share the same shape/grid
        out_profile = nir_profile.copy()
        out_profile["dtype"] = str(aligned_raw.dtype)

        write_singleband_tif(dst_plant_dir / src_path.name, aligned_raw, out_profile)
        debug["aligned_bands"][band_key] = dbg

    # Save debug json
    (dst_plant_dir / "alignment_debug.json").write_text(
        json.dumps(debug, indent=2),
        encoding="utf-8"
    )

    print(f"  Done -> {dst_plant_dir}")


# ------------------------------------------------------------
# Dataset traversal
# ------------------------------------------------------------

def iter_field_dirs(src_root: Path):
    for p in sorted(src_root.iterdir()):
        if p.is_dir():
            if ONLY_FIELD is None or p.name == ONLY_FIELD:
                yield p


def iter_plant_dirs(field_dir: Path):
    for p in sorted(field_dir.iterdir()):
        if p.is_dir() and p.name.lower().startswith("pianta"):
            yield p


def main():
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"SRC_ROOT not found: {SRC_ROOT}")

    DST_ROOT.mkdir(parents=True, exist_ok=True)

    for field_dir in iter_field_dirs(SRC_ROOT):
        print(f"\n=== FIELD: {field_dir.name} ===")
        dst_field_dir = DST_ROOT / field_dir.name
        dst_field_dir.mkdir(parents=True, exist_ok=True)

        for plant_dir in iter_plant_dirs(field_dir):
            dst_plant_dir = dst_field_dir / plant_dir.name
            try:
                process_plant(plant_dir, dst_plant_dir)
            except Exception as e:
                print(f"[ERROR] Failed on {plant_dir}: {e}")

    print("\nAll done.")


if __name__ == "__main__":
    main()