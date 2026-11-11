"""
evaluate.py - final evaluation of the trained model.

1. Validation set (has ground truth): Dice score per image + mean Dice
2. Test set (no ground truth): predicted masks shown on top of the X-rays

Run after main.py:
    python evaluate.py
"""
import os

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from args import get_args
from dataset import KneeDataset
from model import UNet
from utils import dice_score_from_logits, get_device


def predict(model, xray, device):
    # xray: tensor (1, 1, H, W) -> binary mask as numpy (H, W)
    with torch.no_grad():
        logits = model(xray.to(device))
    probs = torch.sigmoid(logits)
    pred = (probs > 0.5).float()
    return logits, pred[0, 0].cpu().numpy()


def overlay(ax, xray, mask, color, title):
    # Show the X-ray in gray and the mask as a transparent color on top
    ax.imshow(xray, cmap="gray")
    colored = np.zeros((*mask.shape, 4))
    colored[mask > 0] = color
    ax.imshow(colored)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def evaluate_validation(model, val_df, device, out_dir):
    loader = DataLoader(KneeDataset(val_df), batch_size=1, shuffle=False)
    rows = []

    for batch in loader:
        logits, pred = predict(model, batch["xray"], device)
        dice = dice_score_from_logits(logits, batch["mask"].to(device))
        rows.append({
            "image": os.path.basename(batch["xray_path"][0]),
            "dice": dice,
            "xray": batch["xray"][0, 0].numpy(),
            "mask": batch["mask"][0, 0].numpy(),
            "pred": pred,
        })

    results = pd.DataFrame([{"image": r["image"], "dice": r["dice"]} for r in rows])
    results.to_csv(os.path.join(out_dir, "val_dice_scores.csv"), index=False)

    print("Dice score per validation image:")
    print(results.to_string(index=False, float_format="%.4f"))
    print(f"\nMean Dice (validation, {len(results)} images): {results['dice'].mean():.4f} "
          f"(std {results['dice'].std():.4f}, min {results['dice'].min():.4f}, max {results['dice'].max():.4f})")

    # Figure: X-ray | ground truth | prediction, for every validation image
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
    for i, r in enumerate(rows):
        axes[i, 0].imshow(r["xray"], cmap="gray")
        axes[i, 0].set_title(r["image"], fontsize=9)
        axes[i, 0].axis("off")
        overlay(axes[i, 1], r["xray"], r["mask"], [0, 1, 0, 0.4], "ground truth")
        overlay(axes[i, 2], r["xray"], r["pred"], [1, 0, 0, 0.4], f"prediction, Dice {r['dice']:.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "val_predictions.png"), dpi=100)
    plt.close(fig)


def predict_test(model, test_df, device, out_dir):
    loader = DataLoader(KneeDataset(test_df), batch_size=1, shuffle=False)
    mask_dir = os.path.join(out_dir, "test_masks")
    os.makedirs(mask_dir, exist_ok=True)

    xrays, preds, names = [], [], []
    for batch in loader:
        _, pred = predict(model, batch["xray"], device)
        name = os.path.basename(batch["xray_path"][0])
        cv2.imwrite(os.path.join(mask_dir, name), (pred * 255).astype(np.uint8))
        xrays.append(batch["xray"][0, 0].numpy())
        preds.append(pred)
        names.append(name)

    # Figure: all test images with the predicted mask on top (5 per row)
    n_cols = 5
    n_rows = int(np.ceil(len(xrays) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, xray, pred, name in zip(axes.flat, xrays, preds, names):
        overlay(ax, xray, pred, [1, 0, 0, 0.4], name)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "test_predictions.png"), dpi=100)
    plt.close(fig)

    print(f"\nTest set: {len(xrays)} predicted masks saved to {mask_dir}")


def main():
    args = get_args()
    device = get_device()

    model = UNet(n_inputs=1, n_classes=1).to(device)
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "best_model.pth"), map_location=device))
    model.eval()

    val_df = pd.read_csv(os.path.join(args.csv_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(args.csv_dir, "test.csv"), keep_default_na=False)

    evaluate_validation(model, val_df, device, args.out_dir)
    predict_test(model, test_df, device, args.out_dir)
    print("Figures saved to:", args.out_dir)


if __name__ == "__main__":
    main()
