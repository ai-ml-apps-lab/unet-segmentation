import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from unet import UNet


if __name__ == "__main__":

    mode = "binary"
    # mode = "multiclass"

    if mode == 'binary':
        class_dict_path = None
        IMAGE_PATH = r"E:\AB\ai_ml_apps_lab_github_2026\5U-Net\dataset\binary_dataset\images\coronavirus-4947340_1920.jpg"
        out_channels = 1

    elif mode == 'multiclass':
        class_dict_path = r"E:\AB\ai_ml_apps_lab_github_2026\5U-Net\dataset\multiclass_dataset\class_dict.csv"
        IMAGE_PATH = r"E:\AB\ai_ml_apps_lab_github_2026\5U-Net\dataset\multiclass_dataset\images\0001TP_009210.png"
        class_df = pd.read_csv(class_dict_path)
        out_channels = len(class_df)

        id_to_color = {}

        for idx, row in class_df.iterrows():
            id_to_color[idx] = (row["r"], row["g"], row["b"])

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Parameters 
    MODEL_PATH = f"./unet_{mode}.pth"

    # Load Model
    model = UNet(in_channels=3, out_channels=out_channels).to(DEVICE)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    model.eval()

    # Image Transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])


    # Load Image
    image = Image.open(IMAGE_PATH).convert("RGB")
    input_image = transform(image)
    input_image = input_image.unsqueeze(0).to(DEVICE)


    # Prediction
    with torch.no_grad():

        output = model(input_image)

        print(output.shape)

        if mode == 'binary':
            output = torch.sigmoid(output)
            output = (output > 0.5).float()

        elif mode == 'multiclass':
            output = torch.argmax(output, dim=1)


    # Visualization
    pred_mask = output.squeeze().cpu().numpy()

    original = input_image.squeeze().cpu().permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(original)
    plt.title("Input Image")

    plt.subplot(1, 2, 2)
    plt.axis("off")
    if mode == 'binary':
        plt.imshow(pred_mask, cmap="gray")

    elif mode == 'multiclass':
        h, w = pred_mask.shape
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in id_to_color.items():
            rgb_mask[pred_mask == class_id] = color

        plt.imshow(rgb_mask)

        # Training:
        # RGB color -> class ID
        # Prediction visualization:
        # class ID -> RGB color
    
    plt.title("Predicted Mask")

    plt.show()