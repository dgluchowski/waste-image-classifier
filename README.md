# Waste Image Classifier

IBM Skills Network coursework project. The model is a VGG16 transfer-learning classifier that labels a waste image as organic (`O`) or recyclable (`R`).

> **Context.** Coursework for **Deep Learning with Keras and Tensorflow** at IBM Skills Network, cleaned up and consolidated for portfolio use.

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

Do not commit model files. The notebook writes these checkpoints. Both are assets on the [`v1.0-models`](https://github.com/dgluchowski/waste-image-classifier/releases/tag/v1.0-models) release, and that release is the only place to get them:

| File | Bytes | Stage |
| --- | --- | --- |
| `O_R_tlearn_vgg16.keras` | 112455527 | Feature extraction |
| `O_R_tlearn_fine_tune_vgg16.keras` | 104061755 | Fine-tuning |

Download URLs:

- https://github.com/dgluchowski/waste-image-classifier/releases/download/v1.0-models/O_R_tlearn_vgg16.keras
- https://github.com/dgluchowski/waste-image-classifier/releases/download/v1.0-models/O_R_tlearn_fine_tune_vgg16.keras

The repository is private, so those URLs need a GitHub session. `gh release download` uses your login:

```bash
gh release download v1.0-models --repo dgluchowski/waste-image-classifier --pattern 'O_R_tlearn_vgg16.keras'
gh release download v1.0-models --repo dgluchowski/waste-image-classifier --pattern 'O_R_tlearn_fine_tune_vgg16.keras'
```

`scripts/evaluate_test.py --download` runs that same `gh release download` for one of the two asset names and then scores the file.

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

Score a `v1.0-models` checkpoint on `data/test`. `--download` fetches that asset into `models/` (`*.keras` is gitignored) and scores it. `--model` scores a file already downloaded from the same release:

```bash
python scripts/evaluate_test.py --download O_R_tlearn_vgg16.keras
python scripts/evaluate_test.py --download O_R_tlearn_fine_tune_vgg16.keras
python scripts/evaluate_test.py --model models/O_R_tlearn_vgg16.keras
python scripts/evaluate_test.py --model models/O_R_tlearn_fine_tune_vgg16.keras
```

The script prints a classification report and writes `outputs/confusion_matrix.csv` and `outputs/confusion_matrix.png`. `outputs/` is gitignored. If the model file or the test images are missing, the script stops and does not invent scores.

## Results

The validation table and the notebook test subset are copied from notebook outputs. The `data/test` table is a separate run of `scripts/evaluate_test.py` on all 200 images in that folder.

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

`scripts/evaluate_test.py --download` scored both `v1.0-models` checkpoints on every image in `data/test` (100 organic, 100 recyclable). Those images are not in git. Figures below are the script's classification report at `digits=4`. Confusion-matrix rows are true labels and columns are predictions, in order O, R.

| Model | Accuracy | O precision | O recall | O f1 | R precision | R recall | R f1 | Support | Confusion matrix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `O_R_tlearn_vgg16.keras` | 0.8400 | 0.8400 | 0.8400 | 0.8400 | 0.8400 | 0.8400 | 0.8400 | 100 + 100 | [[84, 16], [16, 84]] |
| `O_R_tlearn_fine_tune_vgg16.keras` | 0.8500 | 0.8646 | 0.8300 | 0.8469 | 0.8365 | 0.8700 | 0.8529 | 100 + 100 | [[83, 17], [13, 87]] |

For `O_R_tlearn_vgg16.keras`, macro and weighted averages are precision 0.8400, recall 0.8400, and f1 0.8400. For `O_R_tlearn_fine_tune_vgg16.keras`, both averages are precision 0.8506, recall 0.8500, and f1 0.8499.

## Status

The notebook run, the methodology report, the `v1.0-models` release, and the `data/test` scores from `scripts/evaluate_test.py` are in place. The 200 test images are not committed.
