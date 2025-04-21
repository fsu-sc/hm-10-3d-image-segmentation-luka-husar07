import os
import torch
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

def resize_volume(volume, target_shape, is_label=False):
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    order = 0 if is_label else 1  # nearest for labels, linear for images
    return zoom(volume, zoom_factors, order=order)

def normalize(volume):
    volume = np.clip(volume, 0, np.percentile(volume, 99))  # Optional but helpful
    return (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-6)

# Paths
image_dir = "/home/osz09/DATA_SharedClasses/SharedDatasets/MedicalDecathlon/Task02_Heart/imagesTr"
label_dir = "/home/osz09/DATA_SharedClasses/SharedDatasets/MedicalDecathlon/Task02_Heart/labelsTr"
save_dir = "./preprocessed_data"

# Output directories
os.makedirs(save_dir + "/images", exist_ok=True)
os.makedirs(save_dir + "/labels", exist_ok=True)

# Resize target shape (D, H, W)
target_shape = (32, 64, 64)

saved_count = 0
skipped_empty = 0
skipped_resized_empty = 0

for fname in sorted(os.listdir(image_dir)):
    if not fname.endswith(".nii.gz"):
        continue

    print(f"\n📂 Processing {fname}...")

    image_path = os.path.join(image_dir, fname)
    label_path = os.path.join(label_dir, fname)

    image = nib.load(image_path).get_fdata()
    label = nib.load(label_path).get_fdata()

    original_fg = (label > 0).sum()
    print(f"  → Original label foreground voxels: {original_fg}")

    if original_fg < 10:
        print(f"  ⚠️ Skipping {fname} — label is empty or nearly empty.")
        skipped_empty += 1
        continue

    # Resize
    image = resize_volume(image, target_shape=target_shape)
    label = resize_volume(label, target_shape=target_shape, is_label=True)

    resized_fg = (label > 0).sum()
    print(f"  → After resizing: foreground voxels = {resized_fg}")

    if resized_fg < 5:
        print(f"  ⚠️ Skipping {fname} — too little foreground after resizing.")
        skipped_resized_empty += 1
        continue

    # Normalize image
    image = normalize(image)

    # Convert to torch tensors and add channel dim
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [1, D, H, W]
    label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)  # [1, D, H, W]

    # Save
    torch.save(image_tensor, os.path.join(save_dir, "images", fname.replace(".nii.gz", ".pt")))
    torch.save(label_tensor, os.path.join(save_dir, "labels", fname.replace(".nii.gz", ".pt")))

    saved_count += 1

print("\n✅ Done preprocessing.")
print(f"Saved volumes: {saved_count}")
print(f"Skipped (empty labels): {skipped_empty}")
print(f"Skipped (empty after resizing): {skipped_resized_empty}")
