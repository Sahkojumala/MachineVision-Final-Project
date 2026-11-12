"""
dataset.py - read-image functions and the Dataset class.

The Dataset class tells the model which part of the CSV file is X (the X-ray)
and which part is Y (the mask).
"""
import cv2
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from torch.utils.data import Dataset


def read_xray(path):
    # Read as grayscale -> shape (H, W), values 0...255 (uint8)
    xray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)                   # (224, 224), uint8, arvot 0…255

    # PyTorch layers need float numbers between 0 and 1
    xray = xray.astype(np.float32) / 255.0                          # (224, 224), float32, arvot 0…1

    # Add the channel dimension: (H, W) -> (1, H, W)
    xray = xray.reshape((1, *xray.shape))                           # (1, 224, 224)
    return xray


def read_mask(path):
    # Read as grayscale -> background = 0, bone = 255
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Binary mask: bone = 1, background = 0
    mask = (mask > 0).astype(np.float32)                            # 0 ja 255 → 0.0 ja 1.0

    # Add the channel dimension: (H, W) -> (1, H, W)
    mask = mask.reshape((1, *mask.shape))
    return mask


# Augmentation for the training set only.
# Geometric transforms (flip, rotation, shift, scale) are applied to the X-ray AND the mask
# in exactly the same way. ColorJitter changes only the X-ray, because the bone does not move.
train_transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),                  # left knee <-> right knee
    v2.RandomAffine(degrees=10,                      # rotation -10...+10 degrees
                    translate=(0.05, 0.05),          # shift max 5 % of the image size
                    scale=(0.95, 1.05)),             # zoom 95 %...105 %
    v2.ColorJitter(brightness=0.2, contrast=0.2),    # brightness / contrast +-20 %
])


def augment(xray, mask):
    # tv_tensors tell torchvision which tensor is the image and which is the mask.
    # The mask is then rotated with "nearest" interpolation, so it stays binary (0 or 1),
    # and ColorJitter is not applied to the mask.
    xray = tv_tensors.Image(torch.from_numpy(xray))
    mask = tv_tensors.Mask(torch.from_numpy(mask))

    xray, mask = train_transforms(xray, mask)

    return xray.numpy(), mask.numpy()   # Back to numpy, so the output is the same type as without augmentation


class KneeDataset(Dataset):
    def __init__(self, df, augment=False):                                         # muista taulukko
        # df = pandas DataFrame with columns "xrays" and "masks"
        # augment = True only for the training set
        self.df = df    
        self.augment = augment

    def __len__(self):                                              # montako kuvaa
        return len(self.df)

    def __getitem__(self, index):                                   # anna kuvan numero index
        xray_path = self.df["xrays"].iloc[index]                    # "data/xrays/55.png"
        mask_path = self.df["masks"].iloc[index]                    # "data/masks/55.png"

        xray = read_xray(xray_path)
        result = {"xray": xray, "xray_path": xray_path}

        # Test images do not have a mask (the CSV says "None")
        if isinstance(mask_path, str) and mask_path != "None":
            mask = read_mask(mask_path)

            # New random version of the image every time it is loaded (= every epoch)
            if self.augment:
                xray, mask = augment(xray, mask)

            result["xray"] = xray
            result["mask"] = mask

        return result
    