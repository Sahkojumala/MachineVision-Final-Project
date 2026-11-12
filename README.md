# Knee X-ray bone segmentation with U-Net

Course project for *Machine Vision and Convolutional Neural Networks* (2026).
A U-Net is trained to segment one bone from knee X-ray images (224×224, grayscale).

## Data
- `data/xrays/` – 150 X-ray images
- `data/masks/` – 100 manually annotated binary masks (images 0–99)
- `data/CSVs/` – `train.csv` (80), `val.csv` (20), `test.csv` (50, no masks), made with `make_csv.py`

- The data is not included.

## Project structure
| File | Purpose |
|---|---|
| `args.py` | settings: directories, epochs, batch size, learning rate |
| `dataset.py` | read-image functions and the Dataset class |
| `model.py` | U-Net (from [lext/deep-pipeline](https://github.com/lext/deep-pipeline), MIT License) |
| `utils.py` | Dice loss / Dice score and other helper functions |
| `trainer.py` | model initialization, training loop, validation loop |
| `main.py` | data loaders, starts the training |
| `evaluate.py` | Dice score on the validation set, predicted masks on the test set |

## How to run
Run the commands in the project folder:
```bash
python main.py        # training -> outputs/best_model.pth, outputs/learning_curves.png
python evaluate.py    # evaluation -> outputs/val_dice_scores.csv, val_predictions.png, test_predictions.png
```
Settings can be changed on the command line, e.g. `python main.py --epochs 30 --lr 0.0005 --bs 8`.

## Evaluation
The test images have no ground-truth masks, so the Dice score is calculated on the
validation set (20 images). The test set predictions are checked visually
(predicted mask shown on top of the X-ray).

Dice = 2·|prediction ∩ ground truth| / (|prediction| + |ground truth|), predictions thresholded at 0.5.

## License
`model.py` contains code from [deep-pipeline](https://github.com/lext/deep-pipeline)
by Aleksei Tiulpin, released under the MIT License (see `LICENSE`).


## Experiments
One change at a time; all other settings the same (seed 42, Adam lr 1e-3, batch size 16).
Mean Dice = validation set (20 images), from `evaluate.py`.

| # | Change | Epochs | Mean Dice | Min Dice | Notes |
|---|---|---|---|---|---|
| 0 | Baseline: U-Net, Dice loss | 50 | 0.9806 | 0.9158 | holes and small false spots in some test masks |
| 1 | Baseline | 100 | 0.9821 | 0.9201 | best epoch 97, curves flat after ~60 epochs |
| 2 | + augmentation (flip, rotation ±10°, shift, scale, brightness/contrast) | 100 | | | |


Note: runs with the same seed still differ by about ±0.002–0.003 Dice
(GPU computations are not fully deterministic), so smaller differences are not meaningful.
