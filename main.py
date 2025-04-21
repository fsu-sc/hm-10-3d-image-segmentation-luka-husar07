import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import HeartDataset
from torch.utils.data import DataLoader
from mymodels import UNet3D
from training import dice_score
from sklearn.metrics import jaccard_score, precision_score, recall_score
import os

def evaluate_model(model_path, device):
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    val_dataset = HeartDataset("./preprocessed_data/images", "./preprocessed_data/labels")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    all_preds = []
    all_targets = []
    val_dice_total = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            prob = torch.sigmoid(out).squeeze().cpu().numpy()
            binary_pred = (prob > 0.3).astype(np.uint8)
            binary_target = y.squeeze().cpu().numpy().astype(np.uint8)

            all_preds.append(binary_pred.flatten())
            all_targets.append(binary_target.flatten())
            val_dice_total += dice_score(out, y).item()

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)

    jaccard = jaccard_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    avg_dice = val_dice_total / len(val_loader)

    print("\n================= Evaluation Results =================")
    print(f"Dice Score:  {avg_dice:.4f}")
    print(f"Jaccard Index (IoU): {jaccard:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print("======================================================\n")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluating on device:", device)

    model_path = "best_model.pth"
    evaluate_model(model_path, device)
