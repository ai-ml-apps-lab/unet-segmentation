import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class SegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir, image_size=256):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = sorted(os.listdir(image_dir))

        self.transform_image = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_name = self.images[index]

        image_path = os.path.join(self.image_dir, image_name)
        # mask_path = os.path.join(self.mask_dir, image_name)
        base_name = os.path.splitext(image_name)[0]

        mask_name = base_name + ".png"

        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.transform_image(image)
        mask = self.transform_mask(mask)

        # Convert mask to binary
        mask = (mask > 0.5).float()
        # mask = (mask * 255).long()
        # mask = mask.squeeze(0)

        return image, mask