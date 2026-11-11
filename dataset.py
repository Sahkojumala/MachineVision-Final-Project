"""
dataset.py - read-image functions and the Dataset class.

The Dataset class tells the model which part of the CSV file is X (the X-ray)
and which part is Y (the mask).
"""
import cv2
import numpy as np
from torch.utils.data import Dataset


def read_xray(path):
    # Read as grayscale -> shape (H, W), values 0...255 (uint8)
    xray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # PyTorch layers need float numbers between 0 and 1
    xray = xray.astype(np.float32) / 255.0

    # Add the channel dimension: (H, W) -> (1, H, W)
    xray = xray.reshape((1, *xray.shape))
    return xray


def read_mask(path):
    # Read as grayscale -> background = 0, bone = 255
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Binary mask: bone = 1, background = 0
    mask = (mask > 0).astype(np.float32)

    # Add the channel dimension: (H, W) -> (1, H, W)
    mask = mask.reshape((1, *mask.shape))
    return mask


class KneeDataset(Dataset):
    def __init__(self, df):
        # df = pandas DataFrame with columns "xrays" and "masks"
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        xray_path = self.df["xrays"].iloc[index]
        mask_path = self.df["masks"].iloc[index]

        result = {"xray": read_xray(xray_path), "xray_path": xray_path}

        # Test images do not have a mask (the CSV says "None")
        if isinstance(mask_path, str) and mask_path != "None":
            result["mask"] = read_mask(mask_path)

        return result
