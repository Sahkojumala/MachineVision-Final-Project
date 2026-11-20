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
This project uses a U-Net model implementation from [deep-pipeline] (https://github.com/lext/deep-pipeline), originally developed by Aleksei Tiulpin.

Model license: MIT License

The original implementation has been adapted and combined into `model.py` for this project.

The original copyright notice and full MIT License text are included in the `LICENSE` file.

Original source: deep-pipeline U-Net implementation.


## Experiments
One change at a time; all other settings the same (seed 42, Adam lr 1e-3, batch size 16, Dice loss).
Mean / Min Dice = validation set (20 images), calculated per image by `evaluate.py`.

| # | Change | Epochs | Mean Dice | Min Dice | Notes |
|---|---|---|---|---|---|
| 0 | Baseline: U-Net, Dice loss | 50 | 0.9806 | 0.9158 | holes and small false spots in some test masks |
| 1 | Baseline | 100 | 0.9821 | 0.9201 | best epoch 97, curves flat after ~60 epochs |
| 2 | + augmentation | 100 | 0.9822 | 0.9147 | 14/20 images better; unstable epochs 63–70; best epoch 100 |
| 3 | + ReduceLROnPlateau (factor 0.5, patience 5) | 100 | 0.9820 | 0.9145 | lr halved at epochs 77, 87, 96; smooth end of training |

Note: runs with the same seed still differ by about ±0.002–0.003 Dice
(GPU computations are not fully deterministic), so smaller differences are not meaningful.

**Final model:** #3 (`outputs/augment_sched`). It includes all changes, and the end of its training was the most stable
(validation Dice 0.9857–0.9864 in the last 10 epochs).

## Analysis

### 1. The changes did not change the mean Dice
All 100-epoch runs are within 0.9820–0.9822. This difference is smaller than the random variation between runs:
two runs with exactly the same code and seed gave different results (e.g. validation Dice 0.9779 vs 0.9797 at epoch 32),
because GPU computations are not fully deterministic. Differences below about ±0.003 are therefore not meaningful
with only 20 validation images.

### 2. The mean is limited by three difficult images
On 17 of the 20 validation images the Dice is 0.983–0.996. The three weakest images explain most of the error:

| Image | Dice (#1 / #2 / #3) | Reason |
|---|---|---|
| 4.png | 0.920 / 0.915 / 0.915 | **Annotation:** the ground-truth mask stops at a straight line below the top of the image and includes the notch between the condyles. The prediction follows the bone better than the annotation. |
| 45.png | 0.926 / 0.927 / 0.928 | **Annotation:** the ground-truth mask is shifted left/up from the bone. The prediction follows the bone edges. |
| 73.png | 0.955 / 0.949 / 0.938 | **Inverted image** (bone dark, background bright). Only 2 of 150 images are inverted: 3.png (training) and 73.png (validation); none in the test set. The model has seen only one such image, and its prediction leaks into the soft tissue. |

For 4.png and 45.png a better model cannot raise the Dice, because the error is in the ground truth.
These masks were not corrected afterwards, so that the results stay honest.

### 3. Training stability and the learning rate scheduler
In run #2 the validation Dice suddenly dropped from 0.986 to 0.883 (epochs 63–70) and recovered.
Also the training loss went up, so this was not overfitting: the weights jumped away from a good solution,
probably because lr 1e-3 is too large late in training. Saving the best model by validation loss protected the result.

In run #3 the scheduler reduced the lr only at epochs 77, 87 and 96, because the validation loss kept improving slowly
until then. The drop seen in run #2 did not happen in run #3, but the lr was still the same (1e-3) during epochs 63–70,
so **the scheduler did not prevent it** – the drop was random. The visible effect of the scheduler is a smoother end
of training, so the final result does not depend on the exact stopping epoch.

### 4. Test set
The test images have no ground truth, so they were checked visually (`test_predictions.png`).
The predicted masks follow the femur well; no separate false regions are visible in the final model's predictions.

### 5. What could be tried next
- `RandomInvert` augmentation, so that the model sees more inverted images (would mainly help images like 73.png).
- Checking and correcting the annotations (also in the training set), e.g. masks that stop before the image border.
- BCE + Dice loss (pixel-wise BCE in addition to Dice loss).
- Post-processing: keep the largest connected region and fill holes.
- A larger validation set or k-fold cross-validation, to compare changes more reliably than with 20 images.
