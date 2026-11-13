"""
utils.py - helper functions.

dice_loss_from_logits and dice_score_from_logits are from the course (Mitra, 11.11.2026).
"""
import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt


def dice_loss_from_logits(logits, targets, eps: float = 1e-6):
    probs = torch.sigmoid(logits)                               # (B,1,H,W) logitit → todennäköisyydet 0…1
    targets = targets.float()
    dims = (1,2,3)
    intersection = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()

@torch.no_grad()                                                # gradienttia ei tarvita
def dice_score_from_logits(logits, targets, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()                               # make 0 or 1
    targets = targets.float()
    dims = (1,2,3)
    intersection = (preds * targets).sum(dims)
    union = preds.sum(dims) + targets.sum(dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()


def set_seed(seed):
    # Same seed -> same results every time (makes comparisons fair)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_augmentation_examples(dataset, save_path, n_images=3, n_versions=5):
    # Same image loaded several times -> a new random augmentation every time.
    # Red = mask on top of the augmented X-ray: it must still match the bone.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(n_images, n_versions, figsize=(3 * n_versions, 3 * n_images))
    for row in range(n_images):
        for col in range(n_versions):
            sample = dataset[row]
            xray = sample["xray"][0]            # [0] poistaa kanavaulottuvuuden: (1, 224, 224) → (224, 224)
            mask = sample["mask"][0]

            colored = np.zeros((*mask.shape, 4))
            colored[mask > 0] = [1, 0, 0, 0.35]

            ax = axes[row, col]
            ax.imshow(xray, cmap="gray", vmin=0, vmax=1)    # vmin=0, vmax=1 pitää harmaasävyasteikon kiinteänä. Ilman matplotlib venyttäisi kuvan kontrastin automaattisesti, eikä kirkkausmuutos näkyisi kuvissa
            ax.imshow(colored)
            ax.set_title(f"image {row}, version {col + 1}", fontsize=8)
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    print("Augmentation examples saved to:", save_path)


def plot_learning_curves(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="validation")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Dice loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(epochs, history["val_dice"], label="validation", color="green")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("Dice score")
    ax2.set_title("Validation Dice")
    ax2.legend()

    ax3.plot(epochs, history["lr"], color="purple")
    ax3.set_yscale("log")
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("learning rate")
    ax3.set_title("Learning rate")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
