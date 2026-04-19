from __future__ import annotations

import re
import math
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import cv2

import preprocess_segmentation_v2 as segv2


# -----------------------------
# Config
# -----------------------------

# IMPORTANT: edit these for your machine
DATA_ROOT = Path(r"C:\Users\franc\Desktop\thesis_project\data_aligned")
OUT_ROOT = Path(r"C:\Users\franc\Desktop\thesis_project\processed_patches_2\full_dataset")

PATCH_SIZE = 224

# Channels to export
EXPORT_RGB = True
EXPORT_MS5 = True   # BLU, GREEN, RED, RED-EDGE, NIR
EXPORT_VIS = True   # GNDVI, GCI, NDREI, NRI, GI

# Add mask as last channel (recommended)
ADD_MASK_CHANNEL = True

# If a patch contains almost no crown pixels, you can still export it (default True)
# Set to False if you want to drop empty/background-only patches (only do this if labels were also skipped!)
EXPORT_LOW_COVERAGE_PATCHES = True
MIN_MASK_COVERAGE = 0.03  # 3% crown pixels


# DJI mapping 
# DJI_xxx0.jpg -> RGB
# DJI_xxx1.tif -> BLU
# DJI_xxx2.tif -> GREEN
# DJI_xxx3.tif -> RED
# DJI_xxx4.tif -> RED-EDGE
# DJI_xxx5.tif -> NIR
BAND_ORDER = ["BLU", "GREEN", "RED", "RE", "NIR"]


def sanitize_name(name: str) -> str:
    """
    Make a folder-safe field id for output paths.
    """
    s = name.strip()
    s = s.replace("ù", "u").replace("Ù", "U")
    s = s.replace("'", "")
    s = s.replace(" ", "_")
    s = s.replace("-", "_")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()


def compute_selected_vis(
    green_n: np.ndarray,
    red_n: np.ndarray,
    re_n: np.ndarray,
    nir_n: np.ndarray,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Returns HxWx5 float32 stack with:
      GNDVI, GCI, NDREI, NRI, GI
    Inputs are assumed already normalized float32 bands.
    """
    gndvi = (nir_n - green_n) / (nir_n + green_n + eps)
    gci   = (nir_n / (green_n + eps)) - 1.0
    ndrei = (nir_n - re_n) / (nir_n + re_n + eps)
    nri   = (green_n - red_n) / (green_n + red_n + eps)
    gi    = green_n / (red_n + eps)

    # clip unstable ratios
    gndvi = np.clip(gndvi, -1, 1)
    ndrei = np.clip(ndrei, -1, 1)
    nri   = np.clip(nri, -1, 1)

    gci   = np.clip(gci, -10, 10)
    gi    = np.clip(gi, 0, 10)

    return np.stack([gndvi, gci, ndrei, nri, gi], axis=-1).astype(np.float32)


# -----------------------------
# Label reading
# -----------------------------

def read_patch_labels_xlsx(xlsx_path: Path, invalid_report_path: Path | None = None) -> Dict[int, Dict[str, int]]:
    """
    Reads labels from Excel and returns only valid plants (1/4/9 patches).
    Invalid plants are skipped and optionally written to invalid_report_path.
    """
    df = pd.read_excel(xlsx_path, sheet_name=0, header=None)

    pat = re.compile(r"^Pianta(\d+)_([0-9]+)$")
    tmp: Dict[int, Dict[int, int]] = {}

    for _, row in df.iterrows():
        key = row.iloc[0]
        lab = row.iloc[1]
        if not isinstance(key, str):
            continue
        key = key.strip()
        m = pat.match(key)
        if not m:
            continue

        plant_num = int(m.group(1))
        idx = int(m.group(2))

        if pd.isna(lab):
            continue
        lab_int = int(lab)
        if lab_int not in (0, 1):
            continue

        tmp.setdefault(plant_num, {})[idx] = lab_int

    invalid = []

    labels_by_plantnum: Dict[int, Dict[str, int]] = {}
    for plant_num, idx_map in tmp.items():
        k = len(idx_map)
        N = int(round(math.sqrt(k)))

        if N * N != k or N not in (1, 2, 3):
            invalid.append({
                "plant_num": plant_num,
                "num_labels": k,
                "indices_present": ",".join(str(i) for i in sorted(idx_map.keys())),
                "reason": "not_square_1_4_9"
            })
            continue

        out: Dict[str, int] = {}
        for idx, lab_int in idx_map.items():
            if idx < 1 or idx > N * N:
                invalid.append({
                    "plant_num": plant_num,
                    "num_labels": k,
                    "indices_present": ",".join(str(i) for i in sorted(idx_map.keys())),
                    "reason": f"index_out_of_range_for_N={N}"
                })
                out = {}
                break
            idx0 = idx - 1
            r = idx0 // N
            c = idx0 % N
            out[f"r{r}c{c}"] = lab_int

        if not out:
            continue

        expected = {f"r{r}c{c}" for r in range(N) for c in range(N)}
        missing = expected - set(out.keys())
        if missing:
            invalid.append({
                "plant_num": plant_num,
                "num_labels": k,
                "indices_present": ",".join(str(i) for i in sorted(idx_map.keys())),
                "reason": f"missing_cells:{'|'.join(sorted(missing))}"
            })
            continue

        labels_by_plantnum[plant_num] = out

    if invalid_report_path is not None:
        invalid_df = pd.DataFrame(invalid).sort_values(["plant_num"])
        invalid_df.to_csv(invalid_report_path, index=False, encoding="utf-8")

    return labels_by_plantnum


# -----------------------------
# File discovery in each Pianta folder
# -----------------------------

@dataclass
class PlantFiles:
    rgb: Path
    blu: Path
    green: Path
    red: Path
    re: Path
    nir: Path


def find_plant_files(plant_dir: Path) -> PlantFiles:
    """
    Finds DJI_xxx0.(JPG/jpg/jpeg) and DJI_xxx1..5.(TIF/tif)
    Uses the common prefix DJI_####.
    """
    rgb_candidates = (
        list(plant_dir.glob("DJI_*0.JPG")) +
        list(plant_dir.glob("DJI_*0.jpg")) +
        list(plant_dir.glob("DJI_*0.JPEG")) +
        list(plant_dir.glob("DJI_*0.jpeg"))
    )
    if not rgb_candidates:
        raise FileNotFoundError(f"No RGB DJI_*0.jpg in {plant_dir}")

    rgb_path = rgb_candidates[0]
    stem = rgb_path.stem
    if not stem.endswith("0"):
        raise RuntimeError(f"RGB stem does not end with 0: {stem}")

    prefix = stem[:-1]

    def tif_for(digit: int) -> Path:
        cands = list(plant_dir.glob(f"{prefix}{digit}.TIF")) + list(plant_dir.glob(f"{prefix}{digit}.tif"))
        if not cands:
            raise FileNotFoundError(f"Missing {prefix}{digit}.tif in {plant_dir}")
        return cands[0]

    return PlantFiles(
        rgb=rgb_path,
        blu=tif_for(1),
        green=tif_for(2),
        red=tif_for(3),
        re=tif_for(4),
        nir=tif_for(5),
    )


def find_labels_xlsx(field_dir: Path) -> Path:
    xlsx_files = sorted(field_dir.glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx label file found in {field_dir}")
    if len(xlsx_files) > 1:
        print(f"[WARN] Multiple xlsx files found in {field_dir}, using: {xlsx_files[0].name}")
    return xlsx_files[0]


# -----------------------------
# Patch extraction helpers
# -----------------------------

def pad_to_square(arr: np.ndarray, pad_value: float = 0) -> np.ndarray:
    if arr.ndim == 2:
        H, W = arr.shape
    else:
        H, W, _ = arr.shape

    S = max(H, W)
    pad_top = (S - H) // 2
    pad_bottom = S - H - pad_top
    pad_left = (S - W) // 2
    pad_right = S - W - pad_left

    if arr.ndim == 2:
        return np.pad(
            arr,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=pad_value
        )
    else:
        return np.pad(
            arr,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=pad_value
        )


def resize(arr: np.ndarray, out_size: int, is_mask: bool = False) -> np.ndarray:
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(arr.astype(np.float32), (out_size, out_size), interpolation=interp)


def grid_boxes(H: int, W: int, N: int) -> List[Tuple[int, int, int, int]]:
    ys = np.linspace(0, H, N + 1).astype(int)
    xs = np.linspace(0, W, N + 1).astype(int)
    boxes = []
    for r in range(N):
        for c in range(N):
            y0, y1 = ys[r], ys[r + 1]
            x0, x1 = xs[c], xs[c + 1]
            boxes.append((y0, y1, x0, x1))
    return boxes


def infer_N_from_label_map(label_map: Dict[str, int]) -> int:
    k = len(label_map)
    N = int(round(math.sqrt(k)))
    if N * N != k or N not in (1, 2, 3):
        raise ValueError(f"label_map has {k} entries; expected 1/4/9.")
    return N


# -----------------------------
# Export
# -----------------------------

def build_sample_stack(
    rgb_crop: Optional[np.ndarray],
    ms5_crop: Optional[np.ndarray],
    vis_crop: Optional[np.ndarray],
    mask_crop: np.ndarray,
    add_mask_channel: bool
) -> np.ndarray:
    """
    Returns HxWxC float32 stack.
    RGB assumed uint8, normalized to [0,1].
    ms5 assumed float32, normalized already to [0,1].

    Channel order:
      RGB(3) + MS5(5) + VI5(5) + MASK(1) = 14 channels
    """
    chans = []
    if rgb_crop is not None:
        rgb_f = rgb_crop.astype(np.float32) / 255.0
        chans.append(rgb_f)
    if ms5_crop is not None:
        chans.append(ms5_crop.astype(np.float32))
    if vis_crop is not None:
        chans.append(vis_crop.astype(np.float32))
    if add_mask_channel:
        chans.append(mask_crop.astype(np.float32)[..., None])

    if not chans:
        raise ValueError("No channels selected to export (RGB/MS5/VIs/mask).")

    return np.concatenate(chans, axis=-1).astype(np.float32)


def export_one_field(field_dir: Path, field_out_root: Path):
    labels_xlsx = find_labels_xlsx(field_dir)
    date_id = field_dir.name

    print(f"\n=== FIELD: {field_dir.name} ===")
    print(f"Labels file: {labels_xlsx.name}")

    field_out_root.mkdir(parents=True, exist_ok=True)

    labels_by_plantnum = read_patch_labels_xlsx(
        labels_xlsx,
        field_out_root / "invalid_label_plants.csv"
    )

    index_rows = []

    plant_dirs = sorted([p for p in field_dir.iterdir() if p.is_dir() and p.name.lower().startswith("pianta")])

    for plant_dir in plant_dirs:
        try:
            plant_num = int(plant_dir.name.lower().replace("pianta", "").strip())
        except Exception:
            print(f"[SKIP] Cannot parse plant number from folder: {plant_dir.name}")
            continue

        if plant_num not in labels_by_plantnum:
            print(f"[SKIP] No Excel labels for plant {plant_num} ({plant_dir.name})")
            continue

        label_map = labels_by_plantnum[plant_num]
        N = infer_N_from_label_map(label_map)

        try:
            files = find_plant_files(plant_dir)
        except Exception as e:
            print(f"[SKIP] {plant_dir.name}: {e}")
            continue

        rgb = segv2.read_rgb(files.rgb) if EXPORT_RGB else None

        blu = segv2.read_singleband_tif(files.blu)
        green = segv2.read_singleband_tif(files.green)
        red = segv2.read_singleband_tif(files.red)
        re_ = segv2.read_singleband_tif(files.re)
        nir = segv2.read_singleband_tif(files.nir)

        blu_n = segv2.normalize_percentile(blu)
        green_n = segv2.normalize_percentile(green)
        red_n = segv2.normalize_percentile(red)
        re_n = segv2.normalize_percentile(re_)
        nir_n = segv2.normalize_percentile(nir)

        vis5 = None
        if EXPORT_VIS:
            vis5 = compute_selected_vis(
                green_n=green_n,
                red_n=red_n,
                re_n=re_n,
                nir_n=nir_n,
            )

        ms5 = None
        if EXPORT_MS5:
            ms5 = np.stack([blu_n, green_n, red_n, re_n, nir_n], axis=-1).astype(np.float32)

        params = segv2.SegmentationParams(align_nir_to_red=False)
        seg_res, debug = segv2.segment_target_tree_from_red_nir(red=red, nir=nir, params=params)

        mask = seg_res.mask.astype(np.uint8)
        y0, y1, x0, x1 = seg_res.bbox

        mask_c = mask[y0:y1, x0:x1]
        rgb_c = rgb[y0:y1, x0:x1] if rgb is not None else None
        ms5_c = ms5[y0:y1, x0:x1] if ms5 is not None else None
        vis5_c = vis5[y0:y1, x0:x1] if vis5 is not None else None

        Hc, Wc = mask_c.shape
        boxes = grid_boxes(Hc, Wc, N)

        plant_id = f"Pianta{plant_num:03d}"
        sample_dir = field_out_root / plant_id / date_id
        patches_dir = sample_dir / "patches"
        patches_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "field_name": field_dir.name,
            "field_id": sanitize_name(field_dir.name),
            "plant_id": plant_id,
            "plant_num": plant_num,
            "date_id": date_id,
            "pianta_folder": str(plant_dir),
            "bbox": [int(y0), int(y1), int(x0), int(x1)],
            "N": N,
            "crown_area_px": int(mask.sum()),
            "seg_score": float(seg_res.score),
            "is_overlap_likely": bool(seg_res.is_overlap_likely),
            "seg_debug": seg_res.debug,
            "v2_debug_keys": list(debug.keys()),
            "params": params.__dict__,
            "files": {
                "rgb": str(files.rgb),
                "blu": str(files.blu),
                "green": str(files.green),
                "red": str(files.red),
                "re": str(files.re),
                "nir": str(files.nir),
            }
        }
        (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        for idx, (py0, py1, px0, px1) in enumerate(boxes):
            r = idx // N
            c = idx % N
            patch_id = f"r{r}c{c}"

            y = int(label_map[patch_id])

            m_patch = mask_c[py0:py1, px0:px1]
            coverage = float(m_patch.mean()) if m_patch.size else 0.0

            if (not EXPORT_LOW_COVERAGE_PATCHES) and (coverage < MIN_MASK_COVERAGE):
                continue

            rgb_p = rgb_c[py0:py1, px0:px1] if rgb_c is not None else None
            ms5_p = ms5_c[py0:py1, px0:px1] if ms5_c is not None else None
            vis5_p = vis5_c[py0:py1, px0:px1] if vis5_c is not None else None

            if m_patch.size:
                if rgb_p is not None:
                    rgb_p = rgb_p.copy()
                    rgb_p[m_patch == 0] = 0
                if ms5_p is not None:
                    ms5_p = ms5_p.copy()
                    ms5_p[m_patch == 0] = 0
                if vis5_p is not None:
                    vis5_p = vis5_p.copy()
                    vis5_p[m_patch == 0] = 0

            m_sq = pad_to_square(m_patch, pad_value=0)
            m_rs = resize(m_sq, PATCH_SIZE, is_mask=True)
            m_rs = (m_rs > 0.5).astype(np.float32)

            if rgb_p is not None:
                rgb_sq = pad_to_square(rgb_p, pad_value=0)
                rgb_rs = resize(rgb_sq, PATCH_SIZE, is_mask=False)
                rgb_rs = np.clip(rgb_rs, 0, 255).astype(np.float32)
            else:
                rgb_rs = None

            if ms5_p is not None:
                ms_sq = pad_to_square(ms5_p, pad_value=0)
                ms_rs = resize(ms_sq, PATCH_SIZE, is_mask=False)
                ms_rs = np.clip(ms_rs, 0.0, 1.0).astype(np.float32)
            else:
                ms_rs = None

            if vis5_p is not None:
                vis_sq = pad_to_square(vis5_p, pad_value=0)
                vis_rs = resize(vis_sq, PATCH_SIZE, is_mask=False)
                vis_rs = vis_rs.astype(np.float32)
            else:
                vis_rs = None

            x = build_sample_stack(rgb_rs, ms_rs, vis_rs, m_rs, add_mask_channel=ADD_MASK_CHANNEL)

            patch_meta = {
                "field_name": field_dir.name,
                "field_id": sanitize_name(field_dir.name),
                "plant_id": plant_id,
                "date_id": date_id,
                "patch_id": patch_id,
                "N": N,
                "row": r,
                "col": c,
                "crop_box_in_bbox": [int(py0), int(py1), int(px0), int(px1)],
                "mask_coverage": coverage,
                "label": y,
            }

            out_path = patches_dir / f"{patch_id}.npz"
            np.savez_compressed(
                out_path,
                x=x.astype(np.float32),
                y=np.int64(y),
                meta=json.dumps(patch_meta).encode("utf-8"),
            )

            index_rows.append({
                "field_name": field_dir.name,
                "field_id": sanitize_name(field_dir.name),
                "plant_id": plant_id,
                "date_id": date_id,
                "patch_id": patch_id,
                "label": y,
                "N": N,
                "row": r,
                "col": c,
                "mask_coverage": coverage,
                "npz_path": str(out_path),
                "channels": int(x.shape[-1]),
            })

        print(f"[OK] {field_dir.name} | {plant_id} exported (N={N})")

    if index_rows:
        df = pd.DataFrame(index_rows)
        df.to_csv(field_out_root / "dataset_index.csv", index=False, encoding="utf-8")
        print(f"[DONE] Wrote dataset_index.csv with {len(df)} patches -> {field_out_root}")
        return df

    print(f"[DONE] No patches exported for {field_dir.name}")
    return pd.DataFrame()


def export_dataset():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_field_dfs = []

    field_dirs = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()])

    for field_dir in field_dirs:
        field_id = sanitize_name(field_dir.name)
        field_out_root = OUT_ROOT / field_id

        try:
            df_field = export_one_field(field_dir, field_out_root)
            if len(df_field) > 0:
                all_field_dfs.append(df_field)
        except Exception as e:
            print(f"[ERROR] Failed field {field_dir.name}: {e}")

    if all_field_dfs:
        df_all = pd.concat(all_field_dfs, ignore_index=True)
        df_all.to_csv(OUT_ROOT / "dataset_index_full.csv", index=False, encoding="utf-8")
        print(f"\n[GLOBAL DONE] Wrote dataset_index_full.csv with {len(df_all)} patches -> {OUT_ROOT}")
    else:
        print("\n[GLOBAL DONE] No patches exported from any field.")


if __name__ == "__main__":
    export_dataset()