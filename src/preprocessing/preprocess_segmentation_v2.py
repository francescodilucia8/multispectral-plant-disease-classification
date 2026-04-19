# preprocess_segmentation.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np

import rasterio
import cv2

from skimage.morphology import (
    remove_small_objects,
    binary_closing,
    binary_opening,
    binary_dilation,
    disk,
)
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from PIL import Image


# -----------------------------
# Data classes
# -----------------------------

@dataclass
class SegmentationParams:
    # NDVI + normalization
    ndvi_threshold: float = 0.35         # typically 0.2–0.3
    clip_p_low: float = 1.0
    clip_p_high: float = 99.0

    # mask cleanup
    disable_cleanup: bool = False
    min_blob_area: int = 200000
    smooth_radius: int = 3
    open_radius: int = 1

    # alignment
    align_nir_to_red: bool = False
    align_roi_frac: float = 0.7          # use central ROI to estimate shift
    align_min_response: float = 0.05     # below this, alignment is dubious; fall back to no shift

    # touching-crowns split (watershed on distance-transform)
    split_touching: bool = True
    split_min_peak_distance: int = 40
    split_min_peak_abs: float = 10.0

        # forced 2-split on the chosen blob if overlap-likely
    force_split_if_overlap: bool = True
    force_split_n_parts: int = 2          # keep 2, for “couples”
    force_split_min_area_frac: float = 0.10  # ignore tiny pieces (<10% of blob)

    # plausibility scoring weights
    w_center: float = 0.70
    w_area: float = 0.15
    w_shape: float = 0.15

    # center prior (fixes “tiny center_score” issue)
    # center_score = exp(-d^2 / (2*sigma^2)), with sigma = center_sigma_frac * min(H, W)
    center_sigma_frac: float = 0.30

    # expected single-crown area band (pixels) - tune after inspecting
    expected_area_min: int = 250000
    expected_area_max: int = 1000000

    # overlap-likely heuristics
    overlap_area_factor: float = 1.6
    min_solidity: float = 0.85

    # refinement (“separate first, then expand”)
    refine_grow: bool = True
    refine_ndvi_threshold: float = 0.28   # should be <= ndvi_threshold
    refine_iters: int = 140               # growth cap (safety)
    refine_min_area: int = 500            # remove tiny islands in allowed mask

    # debugging
    print_candidates: bool = False
    
    post_smooth_radius: int = 2  # small, safe


@dataclass
class SegmentationResult:
    mask: np.ndarray                    # bool HxW
    bbox: Tuple[int, int, int, int]     # (y0, y1, x0, x1) inclusive
    chosen_label: int
    score: float
    is_overlap_likely: bool
    debug: Dict[str, float]


# -----------------------------
# I/O helpers
# -----------------------------

def read_singleband_tif(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    return arr


def read_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


# -----------------------------
# Normalization + NDVI
# -----------------------------

def normalize_percentile(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    lo = np.percentile(img, p_low)
    hi = np.percentile(img, p_high)
    img = np.clip(img, lo, hi)
    denom = (hi - lo) if (hi - lo) > 1e-8 else 1e-8
    return ((img - lo) / denom).astype(np.float32)


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    denom = nir + red
    denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
    ndvi = (nir - red) / denom
    return ndvi.astype(np.float32)


# -----------------------------
# Alignment: phase correlation (translation)
# -----------------------------

def estimate_translation_phase_corr(
    ref: np.ndarray,
    mov: np.ndarray,
    roi_frac: float = 0.7
) -> Tuple[float, float, float]:
    """
    Estimate translation (dx, dy) to align mov -> ref using cv2.phaseCorrelate on a central ROI.
    Returns (dx, dy, response). Higher response => more reliable.
    """
    assert ref.shape == mov.shape
    H, W = ref.shape

    rh = max(16, int(H * roi_frac))
    rw = max(16, int(W * roi_frac))
    y0 = (H - rh) // 2
    x0 = (W - rw) // 2

    ref_roi = ref[y0:y0 + rh, x0:x0 + rw].astype(np.float32)
    mov_roi = mov[y0:y0 + rh, x0:x0 + rw].astype(np.float32)

    # Hanning window helps reduce edge effects
    win = cv2.createHanningWindow((rw, rh), cv2.CV_32F)
    ref_roi = ref_roi * win
    mov_roi = mov_roi * win

    (dx, dy), response = cv2.phaseCorrelate(ref_roi, mov_roi)
    return float(dx), float(dy), float(response)


def warp_translation(img: np.ndarray, dx: float, dy: float, interp=cv2.INTER_LINEAR) -> np.ndarray:
    H, W = img.shape
    M = np.array([[1, 0, dx],
                  [0, 1, dy]], dtype=np.float32)
    out = cv2.warpAffine(
        img.astype(np.float32),
        M,
        (W, H),
        flags=interp,
        borderMode=cv2.BORDER_REFLECT
    )
    return out.astype(np.float32)


# -----------------------------
# Mask cleanup + splitting + refinement
# -----------------------------

def clean_mask(mask: np.ndarray, params: SegmentationParams) -> np.ndarray:
    """Cleanup of a boolean vegetation mask to produce coherent blobs."""
    m = mask.astype(bool)

    # remove tiny islands early
    m = remove_small_objects(m, min_size=params.min_blob_area)
    m = binary_fill_holes(m)

    # break thin bridges / speckles
    if params.open_radius > 0:
        m = binary_opening(m, footprint=disk(params.open_radius))

    # remove tiny islands again (opening can create small remnants)
    m = remove_small_objects(m, min_size=params.min_blob_area)

    # optionally smooth ragged boundary (closing can also merge nearby crowns!)
    if params.smooth_radius > 0:
        m = binary_closing(m, footprint=disk(params.smooth_radius))

    m = binary_fill_holes(m)
    return m.astype(bool)


def split_touching_crowns(
    mask: np.ndarray,
    min_peak_distance: int = 40,
    min_peak_abs: float = 10.0
) -> np.ndarray:
    """
    Split a connected vegetation mask into multiple crowns using distance-transform watershed.
    Returns an integer label image (0=background, 1..K=crowns).
    """
    m = mask.astype(bool)
    if m.sum() == 0:
        return np.zeros_like(m, dtype=np.int32)

    dist = ndi.distance_transform_edt(m)

    coords = peak_local_max(
        dist,
        labels=m.astype(np.uint8),
        min_distance=int(min_peak_distance),
        threshold_abs=float(min_peak_abs),
        exclude_border=False
    )

    if coords.size == 0:
        return label(m).astype(np.int32)

    markers = np.zeros_like(dist, dtype=np.int32)
    for i, (y, x) in enumerate(coords, start=1):
        markers[int(y), int(x)] = i

    ws = watershed(-dist, markers, mask=m)
    return ws.astype(np.int32)


def grow_seed_into_mask(seed: np.ndarray, allowed: np.ndarray, max_iters: int = 200) -> np.ndarray:
    """
    Geodesic dilation: iteratively dilate 'seed' but restrict it inside 'allowed'.
    This expands a chosen crown back into a more permissive vegetation mask without jumping gaps.
    """
    seed = seed.astype(bool)
    allowed = allowed.astype(bool)

    curr = seed.copy()
    fp = disk(1)

    for _ in range(int(max_iters)):
        dil = binary_dilation(curr, footprint=fp)
        nxt = np.logical_and(dil, allowed)
        if np.array_equal(nxt, curr):
            break
        curr = nxt

    return curr.astype(bool)



from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

def split_blob_into_two_pick_center(blob_mask: np.ndarray,
                                   center_yx: np.ndarray,
                                   min_peak_distance: int,
                                   min_peak_abs: float,
                                   min_area_frac: float = 0.10) -> np.ndarray:
    if blob_mask.sum() == 0:
        return blob_mask

    dist = distance_transform_edt(blob_mask)

    peaks = peak_local_max(
        dist,
        min_distance=min_peak_distance,
        threshold_abs=min_peak_abs,
        labels=blob_mask.astype(np.uint8),
        num_peaks=2
    )

    if peaks.shape[0] < 2:
        ys, xs = np.nonzero(blob_mask)
        pts = np.stack([ys, xs], axis=1).astype(np.int32)
        if len(pts) < 2:
            return blob_mask
        # sample for speed
        if len(pts) > 5000:
            pts = pts[np.random.choice(len(pts), 5000, replace=False)]
        p0 = pts[np.random.randint(len(pts))]
        p1 = pts[np.argmax(np.sum((pts - p0) ** 2, axis=1))]
        p2 = pts[np.argmax(np.sum((pts - p1) ** 2, axis=1))]
        peaks = np.array([p1, p2], dtype=np.int32)

    markers = np.zeros(blob_mask.shape, dtype=np.int32)
    markers[peaks[0, 0], peaks[0, 1]] = 1
    markers[peaks[1, 0], peaks[1, 1]] = 2

    labels = watershed(-dist, markers, mask=blob_mask)

    p1 = (labels == 1)
    p2 = (labels == 2)

    a = float(blob_mask.sum())
    a1 = float(p1.sum())
    a2 = float(p2.sum())
    if a1 < min_area_frac * a or a2 < min_area_frac * a:
        return blob_mask  # reject degenerate split

    def centroid(mask):
        ys, xs = np.nonzero(mask)
        return np.array([ys.mean(), xs.mean()], dtype=np.float32)

    c1 = centroid(p1)
    c2 = centroid(p2)
    d1 = float(np.linalg.norm(c1 - center_yx))
    d2 = float(np.linalg.norm(c2 - center_yx))
    return p1 if d1 <= d2 else p2


# -----------------------------
# Blob features + scoring
# -----------------------------

def blob_shape_features(r) -> Dict[str, float]:
    area = float(r.area)
    convex_area = float(getattr(r, "area_convex", getattr(r, "convex_area", area)))
    solidity = area / max(convex_area, 1.0)

    minr, minc, maxr, maxc = r.bbox
    h = maxr - minr
    w = maxc - minc
    aspect_ratio = (w / max(h, 1))

    per = float(getattr(r, "perimeter", 0.0))
    circularity = (4.0 * np.pi * area) / max(per * per, 1.0)

    return {
        "area": area,
        "solidity": solidity,
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
    }


def score_blob(r, img_center: np.ndarray, img_shape: Tuple[int, int], params: SegmentationParams):
    feats = blob_shape_features(r)

    H, W = img_shape
    diag = np.sqrt(H * H + W * W)

    cy, cx = np.array(r.centroid, dtype=np.float32)

    # normalized distance (0=center, ~1=corner)
    d = float(np.linalg.norm(np.array([cy, cx]) - img_center)) / diag

    # Gaussian center prior
    sigma = params.center_sigma_frac
    center_score = float(np.exp(-(d ** 2) / (2.0 * sigma * sigma)))

    # area score (same as before)
    amin = params.expected_area_min
    amax = params.expected_area_max
    a = feats["area"]

    if amin <= a <= amax:
        area_score = 1.0
    elif a < amin:
        area_score = a / max(amin, 1.0)
    else:
        area_score = amax / max(a, 1.0)

    # shape
    solidity = feats["solidity"]
    ar = feats["aspect_ratio"]
    circ = feats["circularity"]

    ar_penalty = np.exp(-abs(np.log(max(ar, 1e-6))))
    circ_soft = np.tanh(circ)

    shape_score = 0.55 * solidity + 0.30 * ar_penalty + 0.15 * circ_soft
    shape_score = float(np.clip(shape_score, 0.0, 1.0))

    score = (
        params.w_center * center_score +
        params.w_area * area_score +
        params.w_shape * shape_score
    )

    overlap_likely = (a > params.overlap_area_factor * amax) or (solidity < params.min_solidity)

    debug = {
        "center_score": center_score,
        "area_score": float(area_score),
        "shape_score": float(shape_score),
        "area": float(a),
        "solidity": float(solidity),
        "aspect_ratio": float(ar),
        "circularity": float(circ),
        "distance_to_center_norm": float(d),
    }

    return float(score), debug, bool(overlap_likely)

# -----------------------------
# Main: segment target tree with optional NIR->RED alignment
# -----------------------------
'''
def segment_target_tree_from_red_nir(
    red: np.ndarray,
    nir: np.ndarray,
    params: SegmentationParams
) -> Tuple[SegmentationResult, Dict[str, np.ndarray]]:
    """
    Main crown segmentation:
      1) normalize RED/NIR
      2) optional NIR->RED alignment (phase correlation)
      3) NDVI + threshold -> raw vegetation mask
      4) cleanup
      5) optional split touching crowns (distance-transform watershed)
      6) score blobs (strong center prior) and pick best
      7) optional refinement: grow chosen seed within its own watershed component using lower NDVI threshold
      8) recompute bbox and return result + intermediates for visualization
    """

    # -----------------------------
    # Radiometric normalization
    # -----------------------------
    red_n = normalize_percentile(red, params.clip_p_low, params.clip_p_high)
    nir_n = normalize_percentile(nir, params.clip_p_low, params.clip_p_high)

    # NDVI before alignment (for debugging halos)
    ndvi_raw = compute_ndvi(nir_n, red_n)

    # -----------------------------
    # Optional alignment
    # -----------------------------
    dx = dy = resp = 0.0
    nir_aligned = nir_n

    if params.align_nir_to_red:
        dx, dy, resp = estimate_translation_phase_corr(
            red_n, nir_n, roi_frac=params.align_roi_frac
        )
        if resp >= params.align_min_response:
            nir_aligned = warp_translation(nir_n, dx, dy)
        else:
            dx = dy = 0.0  # ignore dubious alignment

    # NDVI after alignment
    ndvi_aligned = compute_ndvi(nir_aligned, red_n)

    # -----------------------------
    # Threshold vegetation
    # -----------------------------
    raw_mask_raw = ndvi_raw > params.ndvi_threshold
    raw_mask_aligned = ndvi_aligned > params.ndvi_threshold

    # -----------------------------
    # Cleanup
    # -----------------------------
    # -----------------------------
    # Cleanup (optional)
    # -----------------------------
    if getattr(params, "disable_cleanup", False):
        mask_clean = raw_mask_aligned.astype(bool)
    else:
        mask_clean = clean_mask(raw_mask_aligned, params)

    # -----------------------------
    # Connected components / splitting
    # -----------------------------
    if getattr(params, "split_touching", False):
        lbl = split_touching_crowns(
            mask_clean,
            min_peak_distance=getattr(params, "split_min_peak_distance", 40),
            min_peak_abs=getattr(params, "split_min_peak_abs", 10.0),
        )
    else:
        lbl = label(mask_clean).astype(np.int32)

    props = regionprops(lbl)
    if len(props) == 0:
        # fallback: try without area cleanup (or with smaller min_size)
        lbl = label(raw_mask_aligned).astype(np.int32)
        props = regionprops(lbl)
        if len(props) == 0:
            raise RuntimeError(...)

    H, W = mask_clean.shape
    center = np.array([H / 2.0, W / 2.0], dtype=np.float32)

    
# -----------------------------
# Score blobs and pick best
# -----------------------------
    best = None
    best_score = -1e9
    best_debug: Dict[str, float] = {}
    best_overlap = False

    # (optional but recommended) ignore tiny watershed shards AFTER splitting
    for r in props:
        if r.area < params.min_blob_area:
            continue

        s, dbg, ov = score_blob(r, center, (H, W), params)

        # guard against NaN/Inf
        if not np.isfinite(s):
            continue

        if s > best_score:
            best_score = s
            best = r
            best_debug = dbg
            best_overlap = ov

    # Fallback: if everything got filtered out / non-finite
    if best is None:
        # pick the largest component >= min_blob_area if possible; else absolute largest
        big = [r for r in props if r.area >= params.min_blob_area]
        best = max(big if big else props, key=lambda rr: rr.area)
        best_score, best_debug, best_overlap = score_blob(best, center, (H, W), params)

    # --- base seed (best label) ---
    seed = (lbl == best.label)
    # Force split the chosen blob into 2 parts and keep the one closest to image center.
    # We accept the split only if both parts are non-trivial (min_area_frac check).
    seed = split_blob_into_two_pick_center(
        seed.astype(bool),
        center.astype(np.float32),
        min_peak_distance=getattr(params, "split_min_peak_distance", 30),
        min_peak_abs=getattr(params, "split_min_peak_abs", 8.0),
        min_area_frac=getattr(params, "force_split_min_area_frac", 0.10),
    )

    # --- merge center-near fragments into seed ---
    center_thresh = getattr(params, "merge_center_thresh", 0.18)
    diag = np.sqrt(H * H + W * W)

    merged = seed.copy()
    for r in props:
        cy, cx = np.array(r.centroid, dtype=np.float32)
        d = float(np.linalg.norm(np.array([cy, cx]) - center)) / diag
        if d < center_thresh:
            merged |= (lbl == r.label)

    seed = merged
    seed_area = int(seed.sum())
    # --- chosen starts from merged seed ---
    chosen = seed.copy()
    
    if getattr(params, "refine_grow", False):
        refine_thr = getattr(params, "refine_ndvi_threshold", params.ndvi_threshold)
        raw_mask_refine = ndvi_aligned > refine_thr

        # NOTE: if you keep "restrict to watershed component", this must match the MERGED seed
        # Otherwise allowed becomes only best.label again and you shrink back.
        allowed = np.logical_and(raw_mask_refine, seed)  # <-- KEY FIX (use merged seed, not lbl==best.label)

        allowed = binary_opening(allowed.astype(bool), footprint=disk(1))
        allowed = remove_small_objects(
            allowed.astype(bool),
            min_size=int(max(getattr(params, "refine_min_area", 0), params.min_blob_area // 4))
        )

        chosen = grow_seed_into_mask(seed=chosen, allowed=allowed, max_iters=int(getattr(params, "refine_iters", 200)))
    chosen = binary_fill_holes(chosen)
    if params.post_smooth_radius > 0:
        chosen = binary_closing(chosen, footprint=disk(params.post_smooth_radius))
        chosen = binary_fill_holes(chosen)
    final_area = int(chosen.sum())
    # -----------------------------
    # Recompute bbox AFTER refinement
    # -----------------------------
    ys, xs = np.where(chosen)
    if ys.size == 0:
        # extremely rare: refinement removed everything; fall back to seed
        chosen = seed.astype(bool)
        ys, xs = np.where(chosen)

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    # -----------------------------
    # Debug info
    # -----------------------------
    best_debug.update({
        "align_dx": float(dx),
        "align_dy": float(dy),
        "align_response": float(resp),
        "ndvi_threshold": float(params.ndvi_threshold),
        "refine_ndvi_threshold": float(getattr(params, "refine_ndvi_threshold", params.ndvi_threshold)),
        "chosen_area": float(chosen.sum()),
        "seed_area": float(seed.sum()),
        "bbox_h": float(y1 - y0 + 1),
        "bbox_w": float(x1 - x0 + 1),
    })

    result = SegmentationResult(
        mask=chosen.astype(bool),
        bbox=(y0, y1, x0, x1),
        chosen_label=int(best.label),
        score=float(best_score),
        is_overlap_likely=bool(best_overlap),
        debug=best_debug
    )

    intermediates = {
        "red_norm": red_n,
        "nir_norm": nir_n,
        "nir_aligned": nir_aligned,
        "ndvi_raw": ndvi_raw,
        "ndvi_aligned": ndvi_aligned,
        "raw_mask_raw": raw_mask_raw.astype(np.uint8),
        "raw_mask_aligned": raw_mask_aligned.astype(np.uint8),
        "clean_mask_aligned": mask_clean.astype(np.uint8),
        "label_image": lbl.astype(np.int32),
        "chosen_seed": seed.astype(np.uint8),
        "chosen_mask": chosen.astype(np.uint8),
    }

    # If refinement is enabled, include the refine allowed region too (useful for debugging)
    if getattr(params, "refine_grow", False):
        intermediates["raw_mask_refine"] = (ndvi_aligned > getattr(params, "refine_ndvi_threshold", params.ndvi_threshold)).astype(np.uint8)
        # 'allowed' is only defined if refine_grow ran; guard it safely:
        try:
            intermediates["refine_allowed"] = allowed.astype(np.uint8)
        except NameError:
            pass

    print("seed_area:", seed.sum())
    print("final_area:", chosen.sum())

    return result, intermediates
'''

def segment_target_tree_from_red_nir(
    red: np.ndarray,
    nir: np.ndarray,
    params: SegmentationParams
) -> Tuple[SegmentationResult, Dict[str, np.ndarray]]:
    """
    Simplest baseline segmentation:
      - ignore NDVI / cleanup / blob selection entirely
      - return a fixed central square crop
      - target square area is approximately 400000 px

    Notes:
      sqrt(500000) ~= 707.10, so we use side = 707
      -> actual square area = 707 * 707 = 499849 px
    """

    # -------------------------------------------------
    # Keep these for compatibility/debug visualization
    # -------------------------------------------------
    red_n = normalize_percentile(red, params.clip_p_low, params.clip_p_high)
    nir_n = normalize_percentile(nir, params.clip_p_low, params.clip_p_high)
    ndvi_raw = compute_ndvi(nir_n, red_n)
    ndvi_aligned = ndvi_raw.copy()
    nir_aligned = nir_n.copy()

    H, W = red.shape

    # -------------------------------------------------
    # Fixed central square
    # -------------------------------------------------
    side = 707  # nearest integer side to sqrt(400000)

    # If the image is smaller than this on one side, clamp
    side = min(side, H, W)

    cy = H // 2
    cx = W // 2
    half = side // 2

    y0 = cy - half
    x0 = cx - half
    y1 = y0 + side
    x1 = x0 + side

    # Safety clamp in case of odd/even border issues
    if y0 < 0:
        y0 = 0
        y1 = side
    if x0 < 0:
        x0 = 0
        x1 = side
    if y1 > H:
        y1 = H
        y0 = H - side
    if x1 > W:
        x1 = W
        x0 = W - side

    # -------------------------------------------------
    # Build mask = central square only
    # -------------------------------------------------
    chosen = np.zeros((H, W), dtype=bool)
    chosen[y0:y1, x0:x1] = True

    # For compatibility with downstream debug/intermediate usage
    seed = chosen.copy()
    raw_mask_raw = chosen.copy()
    raw_mask_aligned = chosen.copy()
    mask_clean = chosen.copy()
    lbl = chosen.astype(np.int32)  # single "component"

    best_debug = {
        "mode": "fixed_center_square",
        "target_area_px": 500000.0,
        "actual_side_px": float(side),
        "actual_area_px": float(chosen.sum()),
        "align_dx": 0.0,
        "align_dy": 0.0,
        "align_response": 0.0,
        "ndvi_threshold": float(getattr(params, "ndvi_threshold", 0.0)),
        "refine_ndvi_threshold": float(getattr(params, "refine_ndvi_threshold", getattr(params, "ndvi_threshold", 0.0))),
        "chosen_area": float(chosen.sum()),
        "seed_area": float(seed.sum()),
        "bbox_h": float(y1 - y0),
        "bbox_w": float(x1 - x0),
    }

    result = SegmentationResult(
        mask=chosen.astype(bool),
        bbox=(int(y0), int(y1), int(x0), int(x1)),   # exclusive upper bounds, good for slicing
        chosen_label=1,
        score=0.0,
        is_overlap_likely=False,
        debug=best_debug
    )

    intermediates = {
        "red_norm": red_n,
        "nir_norm": nir_n,
        "nir_aligned": nir_aligned,
        "ndvi_raw": ndvi_raw,
        "ndvi_aligned": ndvi_aligned,
        "raw_mask_raw": raw_mask_raw.astype(np.uint8),
        "raw_mask_aligned": raw_mask_aligned.astype(np.uint8),
        "clean_mask_aligned": mask_clean.astype(np.uint8),
        "label_image": lbl.astype(np.int32),
        "chosen_seed": seed.astype(np.uint8),
        "chosen_mask": chosen.astype(np.uint8),
    }

    print("seed_area:", seed.sum())
    print("final_area:", chosen.sum())

    return result, intermediates
# -----------------------------
# Apply bbox/mask to images
# -----------------------------

def crop_with_bbox(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    return img[y0:y1 + 1, x0:x1 + 1]


def apply_mask(img: np.ndarray, mask: np.ndarray, fill_value=0) -> np.ndarray:
    if img.ndim == 2:
        out = img.copy()
        out[~mask] = fill_value
        return out
    if img.ndim == 3:
        out = img.copy()
        out[~mask, :] = fill_value
        return out
    raise ValueError("Unsupported img dimensions.")


# -----------------------------
# Visualization
# -----------------------------

def overlay_mask_on_rgb(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Returns an RGB image with mask overlay (green tint)."""
    out = rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    overlay = out.copy()
    overlay[m, 1] = np.clip(overlay[m, 1] + 120, 0, 255)  # add green
    out = (1 - alpha) * out + alpha * overlay
    return out.astype(np.uint8)


def draw_bbox(rgb: np.ndarray, bbox: Tuple[int, int, int, int], color=(255, 0, 0), thickness: int = 2) -> np.ndarray:
    out = rgb.copy()
    y0, y1, x0, x1 = bbox
    cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)  # cv2 uses (x,y)
    return out


def visualize_alignment_and_mask(
    result: SegmentationResult,
    intermediates: Dict[str, np.ndarray],
    rgb: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """
    Creates a debug figure:
      - NDVI raw vs aligned (halo check)
      - raw masks
      - cleaned mask + chosen mask (and seed if available)
      - RGB overlay (if provided)
    """
    import matplotlib.pyplot as plt

    ndvi_raw = intermediates["ndvi_raw"]
    ndvi_al = intermediates["ndvi_aligned"]
    m_raw = intermediates["raw_mask_raw"]
    m_clean = intermediates["clean_mask_aligned"]
    m_chosen = intermediates["chosen_mask"]
    m_seed = intermediates.get("chosen_seed_mask", None)

    fig = plt.figure(figsize=(14, 10))

    ax1 = plt.subplot(2, 3, 1)
    ax1.set_title("NDVI (raw)")
    im1 = ax1.imshow(ndvi_raw, vmin=-0.2, vmax=0.9, cmap="viridis")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.axis("off")

    ax2 = plt.subplot(2, 3, 2)
    ax2.set_title("NDVI (NIR aligned → RED)")
    im2 = ax2.imshow(ndvi_al, vmin=-0.2, vmax=0.9, cmap="viridis")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.axis("off")

    ax3 = plt.subplot(2, 3, 3)
    ax3.set_title(
        f"Alignment shift dx={result.debug.get('align_dx', 0.0):.2f}, dy={result.debug.get('align_dy', 0.0):.2f}\n"
        f"resp={result.debug.get('align_response', 0.0):.3f}"
    )
    diff = np.abs(ndvi_al - ndvi_raw)
    im3 = ax3.imshow(diff, cmap="magma")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    ax3.axis("off")

    ax4 = plt.subplot(2, 3, 4)
    ax4.set_title("Mask from NDVI (raw)")
    ax4.imshow(m_raw, cmap="gray")
    ax4.axis("off")

    ax5 = plt.subplot(2, 3, 5)
    ax5.set_title("Clean mask + chosen (yellow)\nseed (cyan)")
    ax5.imshow(m_clean, cmap="gray")
    ax5.contour(m_chosen.astype(bool), colors="yellow", linewidths=1)
    if m_seed is not None:
        ax5.contour(m_seed.astype(bool), colors="cyan", linewidths=1)
    ax5.axis("off")

    ax6 = plt.subplot(2, 3, 6)
    ax6.set_title(f"Chosen mask + bbox | score={result.score:.3f} | overlap={result.is_overlap_likely}")
    if rgb is not None:
        rgb_overlay = overlay_mask_on_rgb(rgb, result.mask, alpha=0.35)
        rgb_overlay = draw_bbox(rgb_overlay, result.bbox, color=(255, 0, 0), thickness=2)
        ax6.imshow(rgb_overlay)
    else:
        ax6.imshow(m_chosen, cmap="gray")
        y0, y1, x0, x1 = result.bbox
        ax6.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="red", linewidth=2)
    ax6.axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
