# %%
import os
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import glob

def load_nifti(path):
    """Load NIfTI file and return image array + metadata"""
    nii = nib.load(path)
    data = nii.get_fdata()
    header = nii.header
    return data, header

def summarize_volume(volume, header, name):
    print(f"--- {name} ---")
    print(f"Shape: {volume.shape}")
    print(f"Voxel spacing: {header.get_zooms()}")
    print(f"Min: {np.min(volume):.2f} | Max: {np.max(volume):.2f} | Mean: {np.mean(volume):.2f} | Std: {np.std(volume):.2f}")
    print()

def show_slices(volume, title):
    """Show one slice from each orientation"""
    mid_x = volume.shape[0] // 2
    mid_y = volume.shape[1] // 2
    mid_z = volume.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(volume[mid_x, :, :].T, cmap="gray", origin="lower")
    axes[0].set_title("Axial")
    axes[1].imshow(volume[:, mid_y, :].T, cmap="gray", origin="lower")
    axes[1].set_title("Coronal")
    axes[2].imshow(volume[:, :, mid_z].T, cmap="gray", origin="lower")
    axes[2].set_title("Sagittal")
    fig.suptitle(title)
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    plt.show()

def plot_segmentation_distribution(seg_volume):
    labels, counts = np.unique(seg_volume, return_counts=True)
    labels = labels.astype(int)

    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts, tick_label=labels)
    plt.xticks(labels)
    plt.xlabel("Label")
    plt.ylabel("Voxel Count")
    plt.title("Segmentation Volume Distribution")
    plt.tight_layout()
    plt.show()

def analyze_decathlon_dataset(base_dir):
    images_dir = os.path.join(base_dir, "imagesTr")
    labels_dir = os.path.join(base_dir, "labelsTr")

    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.nii*")))
    label_paths = sorted(glob.glob(os.path.join(labels_dir, "*.nii*")))

    print(f"\n Found {len(image_paths)} training images")
    print(f"Label files: {len(label_paths)}\n")

    for i in range(min(3, len(image_paths))):  # analyze a few examples
        img_id = os.path.basename(image_paths[i]).replace(".nii.gz", "")

        # Load image (channel 0 for MRI)
        image, img_header = load_nifti(image_paths[i])
        label_path = os.path.join(labels_dir, f"{img_id}.nii.gz")
        if os.path.exists(label_path):
            label, _ = load_nifti(label_path)
        else:
            print(f"Label for {img_id} not found.")
            label = None

        summarize_volume(image, img_header, f"{img_id} - Image")
        show_slices(image, f"{img_id} - Image")

        if label is not None:
            summarize_volume(label, img_header, f"{img_id} - Label")
            show_slices(label, f"{img_id} - Label")
            plot_segmentation_distribution(label)

if __name__ == "__main__":
    data_dir = "/home/osz09/DATA_SharedClasses/SharedDatasets/MedicalDecathlon/Task02_Heart"
    analyze_decathlon_dataset(data_dir)


 # %%
