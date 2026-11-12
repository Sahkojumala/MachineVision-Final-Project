"""
main.py - loads the CSV files, creates the datasets and data loaders and starts the training.

Run from the project folder:
    python main.py
"""
import os

import pandas as pd
from torch.utils.data import DataLoader

from args import get_args
from dataset import KneeDataset
from trainer import train
from utils import set_seed, save_augmentation_examples


def main():
    args = get_args()
    set_seed(args.seed)

    # 1. Load the CSV files
    train_df = pd.read_csv(os.path.join(args.csv_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(args.csv_dir, "val.csv"))
    print(f"Training images: {len(train_df)} | validation images: {len(val_df)}")

    # 2. Define X and Y for the model (augmentation only for the training set)
    train_dataset = KneeDataset(train_df, augment=args.augment)
    val_dataset = KneeDataset(val_df, augment=False)                        
    # Validaatiodatalle annetaan aina augment=False, jotta tulokset ovat vertailukelpoisia perusversion kanssa.
    print("Augmentation:", "on" if args.augment else "off")

    # Save a picture of augmented samples, to check them before training
    if args.augment:
        save_augmentation_examples(train_dataset, os.path.join(args.out_dir, "augmentation_examples.png"))

    # 3. Data loaders: shuffle the training set, not the validation set
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)

    # 4. Start the training
    train(args, train_loader, val_loader)


if __name__ == "__main__":
    main()
