from dataset import HeartDataset
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from mymodels import UNet3D
import matplotlib.pyplot as plt
import numpy as np

import os

def visualize_prediction(image, prediction, label, slice_idx=None, save_dir="visualizations", epoch=0):
    os.makedirs(save_dir, exist_ok=True)

    image = image.squeeze().cpu().numpy()
    label = label.squeeze().cpu().numpy()
    pred = (torch.sigmoid(prediction).squeeze().cpu().numpy() > 0.3)

    if slice_idx is None:
        # Find slice indices where the label has foreground
        slices_with_fg = np.where(label.sum(axis=(1, 2)) > 0)[0]
        slice_idx = int(np.median(slices_with_fg)) if len(slices_with_fg) > 0 else image.shape[0] // 2


    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image[slice_idx], cmap='gray')
    plt.title("Image")

    plt.subplot(1, 3, 2)
    plt.imshow(label[slice_idx], cmap='gray')
    plt.title("Ground Truth")

    plt.subplot(1, 3, 3)
    plt.imshow(pred[slice_idx], cmap='gray')
    plt.title("Prediction")

    plt.tight_layout()
    # Save the figure
    filename = f"epoch_{epoch:03d}.png" if isinstance(epoch, int) else f"epoch_{epoch}.png"
    plt.savefig(os.path.join(save_dir, filename))

    plt.close() 


# Dice loss (soft version, differentiable)
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice

# Dice score (non-differentiable, for validation)
def dice_score(preds, targets, threshold=0.03):
    preds = (torch.sigmoid(preds) > threshold).bool()
    targets = targets.bool()
    intersection = torch.logical_and(preds, targets).sum().float()
    return (2. * intersection) / (preds.sum() + targets.sum() + 1e-6)

train_dataset = HeartDataset("./preprocessed_data/images", 
                             "./preprocessed_data/labels")

total_size = len(train_dataset)
val_size = int(0.2 * total_size)
train_size = total_size - val_size

train_ds, val_ds = random_split(train_dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

def train(model, train_loader, val_loader, device, num_epochs=50, patience=5):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    #criterion = DiceLoss()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0]).to(device))
    dice = DiceLoss()
    best_val_dice = 0
    patience_counter = 0
    best_model_path = "best_model.pth"
    writer = SummaryWriter(log_dir="runs/left_atrium_experiment")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            #loss = criterion(out, y)
            loss = 0.5 * dice(out, y) + 0.5 * bce(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_dice = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                print("[DEBUG] Label foreground voxels:", (y > 0).sum().item())
                if epoch % 1 == 0 and val_dice == 0:
                    visualize_prediction(x[0], out[0], y[0], epoch=epoch)
                sigmoid_out = torch.sigmoid(out)
                print("[DEBUG] Pred mean:", sigmoid_out.mean().item())
                val_dice += dice_score(out, y).item()
        val_dice /= len(val_loader)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f}")
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Dice/val", val_dice, epoch)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("Saved new best model!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    writer.close()
    print(f"\nTraining complete. Best Val Dice: {best_val_dice:.4f}")
    return best_model_path

if __name__ == "__main__":
    
    batch_size = 1
    num_epochs = 50
    val_split = 0.2
    patience = 30

    #Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("Using device", device)

    #Model
    model = UNet3D(in_channels=1, out_channels=1).to(device)

    #Train
    train(model, train_loader, val_loader, device, num_epochs=num_epochs, patience=patience)




