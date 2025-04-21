import os
import torch
from torch.utils.data import Dataset

class HeartDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        """
        image_dir: path to preprocessed .pt image tensors
        label_dir: path to preprocessed .pt label tensors
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.filenames = sorted([f for f in os.listdir(image_dir) if f.endswith(".pt")])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        image_path = os.path.join(self.image_dir, fname)
        label_path = os.path.join(self.label_dir, fname)

        image = torch.load(image_path)  # shape: [1, D, H, W]
        label = torch.load(label_path)  # shape: [1, D, H, W]

        return image, label

    
if __name__ == "__main__":
    ds = HeartDataset("./preprocessed_data/images", "./preprocessed_data/labels")
    img, lbl = ds[0]
    print("Image shape:", img.shape)
    print("Label unique values:", torch.unique(lbl))

