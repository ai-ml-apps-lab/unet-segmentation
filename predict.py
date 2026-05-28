import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from unet import UNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "./unet_model.pth"

IMAGE_PATH = r"E:\AB\ai_ml_apps_lab_github_2026\5U-Net\dataset\images\coronavirus-4947340_1920.jpg"


# ------------------------
# Load Model
# ------------------------

model = UNet().to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()


# ------------------------
# Image Transform
# ------------------------

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# ------------------------
# Load Image
# ------------------------

image = Image.open(IMAGE_PATH).convert("RGB")

input_image = transform(image)

input_image = input_image.unsqueeze(0).to(DEVICE)


# ------------------------
# Prediction
# ------------------------

with torch.no_grad():

    output = model(input_image)

    output = torch.sigmoid(output)
    output = (output > 0.5).float()

    # output = torch.softmax(output, dim=1)
    # output = torch.argmax(output, dim=1)

# ------------------------
# Visualization
# ------------------------

pred_mask = output.squeeze().cpu().numpy()

original = input_image.squeeze().cpu().permute(1, 2, 0).numpy()

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original)
plt.title("Input Image")

plt.subplot(1, 2, 2)
plt.imshow(pred_mask, cmap="gray")
# plt.imshow(pred_mask.squeeze(), cmap="jet")

plt.title("Predicted Mask")

plt.show()