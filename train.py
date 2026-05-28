import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import Adam

from dataset import SegmentationDataset
from unet import UNet


# ------------------------
# Settings
# ------------------------

IMAGE_DIR = r"E:\AB\ai_ml_apps_lab_github_2026\5U-Net\dataset\images"
MASK_DIR = r"E:\AB\ai_ml_apps_lab_github_2026\5U-Net\dataset\masks"

MODEL_PATH = "./unet_model.pth"

BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------
# Dataset
# ------------------------

dataset = SegmentationDataset(
    IMAGE_DIR,
    MASK_DIR,
    image_size=256
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ------------------------
# Model
# ------------------------

model = UNet(out_channels=1).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
# criterion = nn.CrossEntropyLoss()

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ------------------------
# Training
# ------------------------

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for images, masks in loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        predictions = model(images)

        loss = criterion(predictions, masks)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")


# ------------------------
# Save Model
# ------------------------

torch.save(model.state_dict(), MODEL_PATH)

print("Model saved successfully.")