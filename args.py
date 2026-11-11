"""
args.py - all settings of the pipeline in one place.

Change the values here (or give them on the command line), e.g.
    python main.py --epochs 30 --lr 0.0005
"""
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Knee X-ray bone segmentation with U-Net")

    # Directories
    parser.add_argument("--csv_dir", type=str, default="data/CSVs", help="folder with train/val/test CSV files")
    parser.add_argument("--out_dir", type=str, default="outputs", help="folder for the saved model and figures")

    # Training
    parser.add_argument("--epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument("--bs", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for Adam")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    args = parser.parse_args()
    return args
