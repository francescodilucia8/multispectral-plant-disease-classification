from pathlib import Path
import re

from preprocess_segmentation_v2 import (
    SegmentationParams, read_singleband_tif, read_rgb,
    segment_target_tree_from_red_nir, visualize_alignment_and_mask
)

params = SegmentationParams(
    ndvi_threshold=0.35,
    align_nir_to_red=False,
    align_roi_frac=0.7,
    align_min_response=0.05,
    min_blob_area=150000, 
    expected_area_min=250000,
    expected_area_max=800000,
    w_center=0.45,   # was 0.45
    w_area=0.35,     # was 0.25
    w_shape=0.20,    # was 0.30
    smooth_radius=10,   # disable closing temporarily
    open_radius=8,
    post_smooth_radius=5,
    print_candidates=False,         # set True if you want the blob log
    split_touching=True,
    split_min_peak_distance=60,
    split_min_peak_abs=10.0, #was 10.0
    center_sigma_frac=0.25,
    refine_grow=True,
    refine_ndvi_threshold=0.25,
    refine_iters=250,
    min_solidity=0.90,
)


FIELD_DIR = Path(r"C:\Users\franc\Desktop\thesis_project\data_aligned\Farigliano 3-6-22")
PLANT_DIR = FIELD_DIR / "Pianta 2"

OUT_DIR = Path("debug_vis") / "Carru_22-6-22" / PLANT_DIR.name
OUT_DIR.mkdir(parents=True, exist_ok=True)

def find_band_files(folder: Path):
    """
    Identify by the last digit before extension:
      DJI_xxx0.jpg -> RGB
      DJI_xxx3.tif -> RED
      DJI_xxx5.tif -> NIR
    """
    imgs = list(folder.glob("DJI_*.*"))

    rgb_path = None
    red_path = None
    nir_path = None

    pat = re.compile(r"DJI_(\d+)(\d)\.(jpg|jpeg|tif|tiff)$", re.IGNORECASE)

    for p in imgs:
        m = pat.match(p.name)
        if not m:
            continue
        last_digit = m.group(2)
        ext = m.group(3).lower()

        if last_digit == "0" and ext in ("jpg", "jpeg"):
            rgb_path = p
        elif last_digit == "3" and ext in ("tif", "tiff"):
            red_path = p
        elif last_digit == "5" and ext in ("tif", "tiff"):
            nir_path = p

    if not (rgb_path and red_path and nir_path):
        raise RuntimeError(
            f"Could not find RGB/RED/NIR in {folder}\n"
            f"Found: {[p.name for p in imgs]}\n"
            "Expected DJI_*0.jpg, DJI_*3.tif, DJI_*5.tif"
        )
    return rgb_path, red_path, nir_path

def main():
    if not PLANT_DIR.exists():
        raise FileNotFoundError(f"Plant folder not found: {PLANT_DIR}")

    rgb_path, red_path, nir_path = find_band_files(PLANT_DIR)

    print("Plant:", PLANT_DIR)
    print("RGB:", rgb_path.name)
    print("RED:", red_path.name)
    print("NIR:", nir_path.name)

    red = read_singleband_tif(red_path)
    nir = read_singleband_tif(nir_path)
    rgb = read_rgb(rgb_path)

    print("Shapes | red:", red.shape, "| nir:", nir.shape, "| rgb:", rgb.shape)

    res, inter = segment_target_tree_from_red_nir(red=red, nir=nir, params=params)

    print("bbox:", res.bbox)
    print("alignment:", res.debug["align_dx"], res.debug["align_dy"], "resp:", res.debug["align_response"])

    save_path = OUT_DIR / "alignment_debug.png"
    visualize_alignment_and_mask(
        result=res,
        intermediates=inter,
        rgb=rgb,
        save_path=save_path,
        show=True
    )

    print("Saved:", save_path)

    

if __name__ == "__main__":
    main()