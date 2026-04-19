
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATASET_INDEX = r"C:\Users\franc\Desktop\thesis_project\processed_patches_2\full_dataset\farigliano_3_6_22\dataset_index.csv"

def show_patch(npz_path, label, plant_id, patch_id):
    data = np.load(npz_path)
    x = data["x"]

    print("x.shape:", x.shape)

    rgb = x[:, :, 0:3]

    has_mask = (x.shape[-1] == 14)  # for your current full config
    right_img = x[:, :, -1]
    right_title = "Mask" if has_mask else "Last channel (not mask)"

    print("Right channel min/max/mean:",
          right_img.min(), right_img.max(), right_img.mean())

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].imshow(np.clip(rgb, 0, 1))
    ax[0].set_title(f"{plant_id} {patch_id}\nLabel={label}")
    ax[0].axis("off")

    ax[1].imshow(right_img, cmap="gray", vmin=0, vmax=1)
    ax[1].set_title(right_title)
    ax[1].axis("off")

    plt.show()

def main():
    df = pd.read_csv(DATASET_INDEX)

    print("Total patches:", len(df))
    print("Diseased:", (df.label == 1).sum())
    print("Healthy:", (df.label == 0).sum())

    while True:
        row = df.sample(1).iloc[0]

        show_patch(
            row.npz_path,
            row.label,
            row.plant_id,
            row.patch_id
        )

        inp = input("Press ENTER for next patch (q to quit): ")
        if inp == "q":
            break

if __name__ == "__main__":
    main()