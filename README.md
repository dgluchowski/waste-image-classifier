# Waste Image Classifier

IBM Skills Network coursework project. The model is a VGG16 transfer-learning classifier that labels a waste image as organic (`O`) or recyclable (`R`).

This repository is the coursework project itself. It is not a production sorting system.

## Overview

The notebook loads ImageNet VGG16 with the original classification head removed, freezes the base, and trains a small binary head (two 512-unit ReLU layers, dropout 0.3, one sigmoid unit). A second run unfreezes `block5_conv3` and trains a new head with RMSprop. Both runs save a Keras checkpoint. The written methodology is in [`docs/methodology-report.md`](docs/methodology-report.md). The executed notebook is [`notebooks/waste_classifier_vgg16.ipynb`](notebooks/waste_classifier_vgg16.ipynb).

The notebook downloads the reduced Waste Classification Dataset (1200 images) into `o-vs-r-split/` on the machine where it runs. That download, the extracted images, and the `.keras` checkpoints are gitignored.

## Repository layout

| Path | Contents |
| --- | --- |
| `notebooks/waste_classifier_vgg16.ipynb` | Executed coursework notebook |
| `docs/methodology-report.md` | Methodology write-up |
| `scripts/evaluate_test.py` | Scores a released `.keras` model on `data/test` |
| `requirements.txt` | Packages named by the notebook install cell |
| `data/test/O/` | Organic test images. Not committed. |
| `data/test/R/` | Recyclable test images. Not committed. |

## Models

Do not commit model files. The notebook writes these checkpoints, and both belong on a GitHub Release:

- `O_R_tlearn_vgg16.keras` — feature-extraction stage
- `O_R_tlearn_fine_tune_vgg16.keras` — fine-tuning stage

The Release is not attached yet. The two files are about 104 MB and 112 MB. Which filename is which size is not recorded in the notebook.

Release URL: TBD

## Test images

Put images for a rerun in these folders:

```text
data/test/O/    organic
data/test/R/    recyclable
```

`data/` is gitignored. The notebook itself does not read `data/test`. Its evaluation cell reads `o-vs-r-split/test/O` and `o-vs-r-split/test/R`, and it keeps only the first 50 paths returned by `glob` for each class.

## Setup

The executed notebook kernel is Python 3.11.3 (`language_info.version` in the notebook metadata). The version cell (execution count 3) prints TensorFlow `2.17.0`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`tensorflow`, `scikit-learn`, and `matplotlib` are pinned to the versions in the notebook install cell. `numpy`, `requests`, `tqdm`, and `Pillow` are not pinned there.

## How to run

Train and evaluate inside the notebook (this downloads the dataset and writes `.keras` files in the working directory):

```bash
jupyter notebook notebooks/waste_classifier_vgg16.ipynb
```

Score a model from the Release on `data/test` after the images and the checkpoint are on disk:

```bash
python scripts/evaluate_test.py --model /path/to/O_R_tlearn_vgg16.keras
python scripts/evaluate_test.py --model /path/to/O_R_tlearn_fine_tune_vgg16.keras
```

The script prints a classification report and writes `outputs/confusion_matrix.csv` and `outputs/confusion_matrix.png`. `outputs/` is gitignored. If the model file or the test images are missing, the script stops and does not invent scores.

## Results

Numbers below are copied from notebook outputs. They are not a rerun on `data/test`. Cells that a resume table still needs are `TBD`.

Validation logs use `steps_per_epoch=5`. Execution count 11 prints `len(train_generator)` as 25, and execution counts 7–8 print 800 training images and 200 validation images at batch size 32. Each epoch therefore trains on 160 of 800 images and scores 192 validation images (`200 // 32`). The validation generator also shuffles.

### Validation, from the training logs

| Stage | Checkpoint | Epoch | Accuracy | Val accuracy | Val loss | Where |
| --- | --- | --- | --- | --- | --- | --- |
| Feature extraction | `O_R_tlearn_vgg16.keras` | 10/10 | 0.8438 | 0.8177 | 0.3754 | `model.fit` cell, execution count 20. Lowest `val_loss` in that log. |
| Fine-tuning | `O_R_tlearn_fine_tune_vgg16.keras` | 10/10 | 0.9062 | 0.9062 | 0.2629 | fine-tune `model.fit` cell, execution count 24. Highest `val_accuracy` and lowest `val_loss` in that log. |

Earlier fine-tune epochs in the same cell include validation accuracy 0.8750 (epoch 3), 0.8802 (epoch 7), and 0.8750 (epoch 8). The final logged validation accuracy is 0.9062.

Both stages log `learning_rate: 1.0000e-04` on epoch 1. A `LearningRateScheduler` is attached in both fits. The feature-extraction compile step sets Adam at `1e-5` (execution count 18), and the fine-tune compile step sets RMSprop at `1e-4` (execution count 24). The logged schedule starts at `1e-4` for both.

### Notebook test subset, from the saved checkpoints

Execution count 29 loads the two `.keras` files and calls `sklearn.metrics.classification_report` on 100 images: the first 50 `glob` matches in `o-vs-r-split/test/O` and the first 50 in `o-vs-r-split/test/R`.

| Model | Accuracy | O precision | O recall | O f1 | R precision | R recall | R f1 | Support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Extract Features Model | 0.77 | 0.75 | 0.80 | 0.78 | 0.79 | 0.74 | 0.76 | 50 + 50 |
| Fine-Tuned Model | 0.79 | 0.78 | 0.80 | 0.79 | 0.80 | 0.78 | 0.79 | 50 + 50 |

Macro and weighted averages in that output match the accuracy for each model (0.77 and 0.79). The notebook does not print a confusion matrix.

### Released model on `data/test`

Not run. `scripts/evaluate_test.py` has no model file and no images in this repository.

| Model | Accuracy | Precision | Recall | F1 | Confusion matrix |
| --- | --- | --- | --- | --- | --- |
| `O_R_tlearn_vgg16.keras` on `data/test` | TBD | TBD | TBD | TBD | TBD |
| `O_R_tlearn_fine_tune_vgg16.keras` on `data/test` | TBD | TBD | TBD | TBD | TBD |

## Status

The notebook run and the methodology report are imported. A GitHub Release for the two checkpoints, images under `data/test/O` and `data/test/R`, and a confusion matrix from `scripts/evaluate_test.py` are still required before the TBD cells above can be replaced.
