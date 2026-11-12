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
