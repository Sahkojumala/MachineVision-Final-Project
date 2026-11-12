"""
trainer.py - model initialization, training loop and validation loop.
"""
import os

import torch
import torch.optim as optim

from model import UNet
from utils import dice_loss_from_logits, dice_score_from_logits, get_device, plot_learning_curves


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in loader:                                # 5 erää
        xrays = batch["xray"].to(device)                #  (16,1,224,224) GPU:lle
        masks = batch["mask"].to(device)                

        optimizer.zero_grad()                           # reset the gradients
        logits = model(xrays)                           # forward pass   (16,1,224,224)
        loss = dice_loss_from_logits(logits, masks)     # calculate the loss
        loss.backward()                                 # backpropagation  # gradientit jokaiselle painolle
        optimizer.step()                                # update weights

        total_loss += loss.item()                       # tensor to int

    # Average loss of all batches = loss of this epoch
    return total_loss / len(loader)                     # epochin loss = erien keskiarvo


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for batch in loader:
            xrays = batch["xray"].to(device)
            masks = batch["mask"].to(device)

            logits = model(xrays)
            total_loss += dice_loss_from_logits(logits, masks).item()
            total_dice += dice_score_from_logits(logits, masks)

    return total_loss / len(loader), total_dice / len(loader)


# main funktion
def train(args, train_loader, val_loader):
    device = get_device()
    print("Device:", device)

    # Model initialization: 1 input channel (grayscale), 1 output channel (bone)
    model = UNet(n_inputs=1, n_classes=1).to(device)                    # mallin alustus ja siirto GPU:lle
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    best_model_path = os.path.join(args.out_dir, "best_model.pth")

    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_dice = validate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        text = f"Epoch {epoch:3d}/{args.epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val Dice {val_dice:.4f}"

        # Save the model only when the validation loss is the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            text += "  <- best, saved"
        print(text)

    plot_learning_curves(history, os.path.join(args.out_dir, "learning_curves.png"))
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print("Best model saved to:", best_model_path)
    return history
