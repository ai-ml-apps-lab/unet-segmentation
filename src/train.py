import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from dataset import SegmentationDataset
from unet import UNet


if __name__ == "__main__":

    mode = "binary"
    # mode = "multiclass"

    if mode == 'binary':
        class_dict_path = None
        IMAGE_DIR = r"E:\AB\ai_ml_apps_lab_github_2026\5U-Net\dataset\binary_dataset\images"
        MASK_DIR = r"E:\AB\ai_ml_apps_lab_github_2026\5U-Net\dataset\binary_dataset\masks"
        criterion = nn.BCEWithLogitsLoss()
        out_channels = 1

    elif mode == 'multiclass':
        class_dict_path = r"E:\AB\ai_ml_apps_lab_github_2026\5U-Net\dataset\multiclass_dataset\class_dict.csv"
        IMAGE_DIR = r"E:\AB\ai_ml_apps_lab_github_2026\5U-Net\dataset\multiclass_dataset\images"
        MASK_DIR = r"E:\AB\ai_ml_apps_lab_github_2026\5U-Net\dataset\multiclass_dataset\masks"
        criterion = nn.CrossEntropyLoss()
        class_df = pd.read_csv(class_dict_path)
        out_channels = len(class_df)


    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Parameters 
    MODEL_PATH = f"./unet_{mode}.pth"
    BATCH_SIZE = 2 # for smaller datasets or limited GPU memory
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    train_ratio = 0.8

    # Dataset
    dataset = SegmentationDataset(
        IMAGE_DIR,
        MASK_DIR,
        class_dict_path = class_dict_path if mode == "multiclass" else None,
        image_size=256
    )


    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Model
    model = UNet(in_channels=3, out_channels=out_channels).to(DEVICE)


    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # Training
    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0

        for images, masks in train_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            predictions = model(images)
            loss = criterion(predictions, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                predictions = model(images)
                loss = criterion(predictions, masks)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # print(predictions.shape)
        # print(masks.shape)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")


    # Save Model
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved successfully.")
