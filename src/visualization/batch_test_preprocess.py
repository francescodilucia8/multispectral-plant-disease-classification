# batch_test_preprocess.py


from __future__ import annotations

import argparse
import csv
import re
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from preprocess_segmentation_v2 import (
    SegmentationParams,
    read_singleband_tif,
    read_rgb,
    segment_target_tree_from_red_nir,
    visualize_alignment_and_mask,
    crop_with_bbox,
    apply_mask,
)

# -----------------------------
# Dataset file mapping helpers
# -----------------------------



# Matches DJI_0010.JPG / DJI_0013.TIF / DJI_1305.TIF etc.
DJI_RE = re.compile(r"DJI_(\d{4})\.(JPG|jpg|TIF|tif)$")

def find_band_files(plant_dir: Path) -> Dict[str, Path]:
    """
    Carrù naming rule (your dataset):
      DJI_xxx0.jpg -> RGB
      DJI_xxx3.tif -> RED
      DJI_xxx5.tif -> NIR
    where xxx are 3 digits and the last digit is the band code.
    Example: DJI_0010.JPG, DJI_0013.TIF, DJI_0015.TIF
    """
    rgb = None
    red = None
    nir = None

    for p in plant_dir.iterdir():
        if not p.is_file():
            continue
        m = DJI_RE.match(p.name)
        if not m:
            continue

        code4 = m.group(1)          # e.g. "0010"
        ext = m.group(2).lower()    # "jpg" or "tif"
        band_digit = int(code4[-1]) # last digit: 0/1/2/3/4/5

        if band_digit == 0 and ext == "jpg":
            rgb = p
        elif band_digit == 3 and ext == "tif":
            red = p
        elif band_digit == 5 and ext == "tif":
            nir = p

    missing = [k for k, v in (("rgb", rgb), ("red", red), ("nir", nir)) if v is None]
    if missing:
        # Extra debug: list DJI_* files we saw
        seen = sorted([x.name for x in plant_dir.iterdir() if x.is_file() and x.name.startswith("DJI_")])
        raise RuntimeError(f"Missing files in {plant_dir.name}: {missing}. Seen: {seen}")

    return {"rgb": rgb, "red": red, "nir": nir}

def save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.dtype != np.uint8:
        # clamp if float
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255)


# -----------------------------
# Flag logic
# -----------------------------

def compute_flags(
    final_area: int,
    score: float,
    center_score: Optional[float],
    align_resp: Optional[float],
    bbox: Tuple[int, int, int, int],
    H: int,
    W: int,
    args,
) -> Tuple[bool, str]:
    """
    Returns (flagged, reason_string).
    """
    reasons = []

    # area-based
    if final_area < args.flag_area_min:
        reasons.append(f"area<{args.flag_area_min}")
    if final_area > args.flag_area_max:
        reasons.append(f"area>{args.flag_area_max}")

    # bbox sanity: too thin / too huge
    y0, y1, x0, x1 = bbox
    bh = y1 - y0 + 1
    bw = x1 - x0 + 1
    if bh < args.flag_bbox_min_side or bw < args.flag_bbox_min_side:
        reasons.append(f"bbox_small<{args.flag_bbox_min_side}")
    if bh > int(args.flag_bbox_max_frac * H) or bw > int(args.flag_bbox_max_frac * W):
        reasons.append(f"bbox_large>{args.flag_bbox_max_frac:.2f}*img")

    # center score (if available)
    if center_score is not None and center_score < args.flag_center_score_min:
        reasons.append(f"center<{args.flag_center_score_min}")

    # alignment response
    if align_resp is not None and align_resp < args.flag_align_resp_min:
        reasons.append(f"align_resp<{args.flag_align_resp_min}")

    # overlap heuristic
    if args.flag_overlap and args.flag_overlap is True:
        # overlap flag will be appended by caller if needed
        pass

    flagged = len(reasons) > 0
    return flagged, ";".join(reasons)


# -----------------------------
# Main batch runner
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field-dir", type=str, required=True, help="Path to field folder, containing 'Pianta *' folders.")
    ap.add_argument("--out-dir", type=str, required=True, help="Where to write outputs.")
    ap.add_argument("--csv-name", type=str, default="summary.csv")

    # flag thresholds (tune as you like)
    ap.add_argument("--flag-area-min", type=int, default=80_000)
    ap.add_argument("--flag-area-max", type=int, default=450_000)
    ap.add_argument("--flag-center-score-min", type=float, default=0.20)
    ap.add_argument("--flag-align-resp-min", type=float, default=0.08)
    ap.add_argument("--flag-bbox-min-side", type=int, default=200)
    ap.add_argument("--flag-bbox-max-frac", type=float, default=0.95)
    ap.add_argument("--flag-overlap", action="store_true", help="Also flag if result.is_overlap_likely is True")

    # segmentation params (defaults you can override)
    ap.add_argument("--ndvi-threshold", type=float, default=0.30)
    ap.add_argument("--refine-ndvi-threshold", type=float, default=0.25)
    ap.add_argument("--refine-iters", type=int, default=200)

    ap.add_argument("--min-blob-area", type=int, default=60_000)
    ap.add_argument("--open-radius", type=int, default=1)
    ap.add_argument("--smooth-radius", type=int, default=0)

    ap.add_argument("--split-touching", action="store_true", default=True)
    ap.add_argument("--split-min-peak-distance", type=int, default=60)
    ap.add_argument("--split-min-peak-abs", type=float, default=18.0)

    ap.add_argument("--w-center", type=float, default=0.60)
    ap.add_argument("--w-area", type=float, default=0.20)
    ap.add_argument("--w-shape", type=float, default=0.20)
    ap.add_argument("--center-sigma-frac", type=float, default=0.25)

    # alignment
    ap.add_argument("--align", action="store_true", default=False)
    ap.add_argument("--align-roi-frac", type=float, default=0.7)
    ap.add_argument("--align-min-response", type=float, default=0.05)

    # output control
    ap.add_argument("--save-debug-on-error", action="store_true", default=True)
    ap.add_argument("--max-plants", type=int, default=0, help="0 = all; otherwise process only first N plants (sorted).")

    args = ap.parse_args()

    field_dir = Path(args.field_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / args.csv_name
    debug_dir = out_dir / "debug_flagged"
    perplant_dir = out_dir / "per_plant"

    # build params
    params = SegmentationParams(
        ndvi_threshold=args.ndvi_threshold,
        align_nir_to_red=args.align,
        align_roi_frac=args.align_roi_frac,
        align_min_response=args.align_min_response,

        min_blob_area=args.min_blob_area,
        open_radius=args.open_radius,
        smooth_radius=args.smooth_radius,

        split_touching=args.split_touching,
        split_min_peak_distance=args.split_min_peak_distance,
        split_min_peak_abs=args.split_min_peak_abs,

        w_center=args.w_center,
        w_area=args.w_area,
        w_shape=args.w_shape,
        center_sigma_frac=args.center_sigma_frac,

        refine_grow=True,
        refine_ndvi_threshold=args.refine_ndvi_threshold,
        refine_iters=args.refine_iters,

        print_candidates=False,
    )

    plant_dirs = sorted([p for p in field_dir.iterdir() if p.is_dir() and p.name.lower().startswith("pianta")])
    if args.max_plants and args.max_plants > 0:
        plant_dirs = plant_dirs[: args.max_plants]

    # CSV header
    header = [
        "plant_name",
        "plant_dir",
        "rgb_path",
        "red_path",
        "nir_path",
        "status",
        "error",

        "bbox_y0", "bbox_y1", "bbox_x0", "bbox_x1",
        "seed_area", "final_area",
        "score",
        "overlap_likely",

        "center_score",
        "distance_to_center_norm",

        "align_dx", "align_dy", "align_response",

        "flagged",
        "flag_reasons",

        "debug_png",
        "mask_png",
        "rgb_crop_png",
        "rgb_masked_png",
        "rgb_crop_masked_png",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=header)
        writer.writeheader()

        for i, plant_dir in enumerate(plant_dirs, start=1):
            row = {k: "" for k in header}
            row["plant_name"] = plant_dir.name
            row["plant_dir"] = str(plant_dir)

            print(f"[{i}/{len(plant_dirs)}] {plant_dir.name}")

            try:
                files = find_band_files(plant_dir)
                row["rgb_path"] = str(files["rgb"])
                row["red_path"] = str(files["red"])
                row["nir_path"] = str(files["nir"])

                red = read_singleband_tif(files["red"])
                nir = read_singleband_tif(files["nir"])
                rgb = read_rgb(files["rgb"])

                H, W = red.shape
                res, inter = segment_target_tree_from_red_nir(red=red, nir=nir, params=params)

                # areas
                seed_area = int(res.debug.get("seed_area", 0))
                final_area = int(res.mask.sum())

                row["status"] = "ok"
                y0, y1, x0, x1 = res.bbox
                row["bbox_y0"], row["bbox_y1"], row["bbox_x0"], row["bbox_x1"] = y0, y1, x0, x1
                row["seed_area"] = seed_area
                row["final_area"] = final_area
                row["score"] = float(res.score)
                row["overlap_likely"] = bool(res.is_overlap_likely)

                row["center_score"] = res.debug.get("center_score", "")
                row["distance_to_center_norm"] = res.debug.get("distance_to_center_norm", "")

                row["align_dx"] = res.debug.get("align_dx", "")
                row["align_dy"] = res.debug.get("align_dy", "")
                row["align_response"] = res.debug.get("align_response", "")

                # save per-plant outputs
                plant_out = perplant_dir / plant_dir.name
                plant_out.mkdir(parents=True, exist_ok=True)

                mask_png = plant_out / "mask.png"
                rgb_crop_png = plant_out / "rgb_crop.png"
                rgb_masked_png = plant_out / "rgb_masked.png"
                rgb_crop_masked_png = plant_out / "rgb_crop_masked.png"

                mask_u8 = mask_to_uint8(res.mask)
                save_png(mask_u8, mask_png)

                rgb_crop = crop_with_bbox(rgb, res.bbox)
                save_png(rgb_crop, rgb_crop_png)

                rgb_masked = apply_mask(rgb, res.mask, fill_value=0)
                save_png(rgb_masked, rgb_masked_png)

                rgb_crop_masked = apply_mask(rgb_crop, crop_with_bbox(res.mask.astype(np.uint8), res.bbox).astype(bool), fill_value=0)
                save_png(rgb_crop_masked, rgb_crop_masked_png)

                row["mask_png"] = str(mask_png)
                row["rgb_crop_png"] = str(rgb_crop_png)
                row["rgb_masked_png"] = str(rgb_masked_png)
                row["rgb_crop_masked_png"] = str(rgb_crop_masked_png)

                # flag logic
                center_score = None
                try:
                    cs = res.debug.get("center_score", None)
                    center_score = float(cs) if cs is not None and cs != "" else None
                except Exception:
                    center_score = None

                align_resp = None
                try:
                    ar = res.debug.get("align_response", None)
                    align_resp = float(ar) if ar is not None and ar != "" else None
                except Exception:
                    align_resp = None

                flagged, reasons = compute_flags(
                    final_area=final_area,
                    score=float(res.score),
                    center_score=center_score,
                    align_resp=align_resp,
                    bbox=res.bbox,
                    H=H, W=W,
                    args=args,
                )

                if args.flag_overlap and res.is_overlap_likely:
                    flagged = True
                    reasons = (reasons + ";" if reasons else "") + "overlap_likely"

                row["flagged"] = bool(flagged)
                row["flag_reasons"] = reasons

                # debug only if flagged
                if flagged:
                    dbg_path = debug_dir / plant_dir.name / "alignment_debug.png"
                    dbg_path.parent.mkdir(parents=True, exist_ok=True)
                    visualize_alignment_and_mask(
                        result=res,
                        intermediates=inter,
                        rgb=rgb,
                        save_path=dbg_path,
                        show=False,
                    )
                    row["debug_png"] = str(dbg_path)

            except Exception as e:
                row["status"] = "error"
                row["error"] = f"{type(e).__name__}: {e}"
                print("  ERROR:", row["error"])

                # Try to save debug if we can (only if segmentation reached intermediates)
                if args.save_debug_on_error:
                    try:
                        dbg_path = debug_dir / plant_dir.name / "error_debug.png"
                        dbg_path.parent.mkdir(parents=True, exist_ok=True)

                        # If we have rgb/red/nir loaded and inter exists, visualize. Otherwise skip.
                        if "rgb" in locals() and "inter" in locals() and "res" in locals():
                            visualize_alignment_and_mask(
                                result=res,
                                intermediates=inter,
                                rgb=rgb,
                                save_path=dbg_path,
                                show=False,
                            )
                            row["debug_png"] = str(dbg_path)
                    except Exception:
                        pass

                # If you want full stack traces in console:
                # traceback.print_exc()

            writer.writerow(row)

    print(f"\nDone. Wrote CSV: {csv_path}")
    print(f"Per-plant outputs: {perplant_dir}")
    print(f"Flagged debug outputs: {debug_dir}")


if __name__ == "__main__":
    main()