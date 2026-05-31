import os
from PIL import Image
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir, class_dict_path = None, image_size=256):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.class_dict_path = class_dict_path
        self.image_size = image_size

        self.images = sorted(os.listdir(image_dir))

        self.transform_image = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])

        if self.class_dict_path is None:
            self.transform_mask = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor()
            ])

        if self.class_dict_path is not None:
            self.class_dict = pd.read_csv(self.class_dict_path)

            self.color_map = {}

            for idx, row in self.class_dict.iterrows():
                self.color_map[(row["r"], row["g"], row["b"])] = idx


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
        if self.class_dict_path is None:
            mask = Image.open(mask_path).convert("L")
        else:
            mask = Image.open(mask_path).convert("RGB")
            mask = mask.resize((self.image_size, self.image_size))
            mask = np.array(mask)

        image = self.transform_image(image)
        if self.class_dict_path is None:
            mask = self.transform_mask(mask)


        if self.class_dict_path is not None:
            # Convert mask to multiclass labels
            label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)

            for color, class_id in self.color_map.items():
                matches = np.all(mask == color, axis=-1) #(H, W, 3)
                label_mask[matches] = class_id

            # print(label_mask.shape)
            mask = torch.tensor(label_mask, dtype=torch.long)

        else:
            # Convert mask to binary
            mask = (mask > 0.5).float()

        return image, mask