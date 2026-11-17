# Waste Image Classifier

VGG16 transfer-learning classifier that labels a waste image as organic (`O`) or recyclable (`R`). This started as coursework for Deep Learning with Keras and Tensorflow at IBM Skills Network.

## Overview

ImageNet VGG16 is loaded with the original classification head removed. The base is frozen, and a small binary head is trained (two 512-unit ReLU layers, dropout 0.3, one sigmoid unit). A second run unfreezes `block5_conv3` and trains a new head with RMSprop. Both runs save a Keras checkpoint. The written methodology is in [`docs/methodology-report.md`](docs/methodology-report.md).

The training run used the reduced Waste Classification Dataset (1200 images) in `o-vs-r-split/`. That download, the extracted images, and the `.keras` checkpoints are gitignored. `scripts/evaluate_test.py` scores a released checkpoint. See [How to run](#how-to-run): the scripts do not reproduce training.

## Repository layout

| Path | Contents |
| --- | --- |
| `docs/methodology-report.md` | Methodology write-up |
| `scripts/evaluate_test.py` | Scores a released `.keras` model on `data/test` |
| `requirements.txt` | Packages from the training environment |
| `data/test/O/` | Organic test images. Not committed. |
| `data/test/R/` | Recyclable test images. Not committed. |

## Models

Do not commit model files. The training run wrote these checkpoints. Both are assets on the [`v1.0-models`](https://github.com/dgluchowski/waste-image-classifier/releases/tag/v1.0-models) release, and that release is the only place to get them:

| File | Bytes | Stage |
| --- | --- | --- |
| `O_R_tlearn_vgg16.keras` | 112455527 | Feature extraction |
| `O_R_tlearn_fine_tune_vgg16.keras` | 104061755 | Fine-tuning |

Download URLs:

- https://github.com/dgluchowski/waste-image-classifier/releases/download/v1.0-models/O_R_tlearn_vgg16.keras
- https://github.com/dgluchowski/waste-image-classifier/releases/download/v1.0-models/O_R_tlearn_fine_tune_vgg16.keras

Those release URLs are public. You can download in the browser, with `curl -L`, or with `gh`:

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

`data/` is gitignored. `scripts/evaluate_test.py` reads every image in `data/test/O` and `data/test/R`. The 0.77 and 0.79 scores below are a different subset: the first 50 paths in `o-vs-r-split/test/O` and the first 50 in `o-vs-r-split/test/R`.

## Setup

The `v1.0-models` checkpoints were produced with Python 3.11.3 and TensorFlow 2.17.0.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`tensorflow`, `scikit-learn`, and `matplotlib` are pinned to the versions from that training environment. `numpy`, `requests`, `tqdm`, and `Pillow` are not pinned.

## How to run

`scripts/` does not reproduce training. The only script scores a checkpoint that already exists. The trained files are the `v1.0-models` release assets listed above.

Score a `v1.0-models` checkpoint on `data/test`. `--download` fetches that asset into `models/` (`*.keras` is gitignored) and scores it. `--model` scores a file already downloaded from the same release:

```bash
python scripts/evaluate_test.py --download O_R_tlearn_vgg16.keras
python scripts/evaluate_test.py --download O_R_tlearn_fine_tune_vgg16.keras
python scripts/evaluate_test.py --model models/O_R_tlearn_vgg16.keras
python scripts/evaluate_test.py --model models/O_R_tlearn_fine_tune_vgg16.keras
```

The script prints a classification report and writes `outputs/confusion_matrix.csv` and `outputs/confusion_matrix.png`. `outputs/` is gitignored. If the model file or the test images are missing, the script stops and does not invent scores.

## Results

The validation table and the 100-image test subset below were recorded from the training run that produced the `v1.0-models` checkpoints. Those logs are not in this repository. The `data/test` table is a separate run of `scripts/evaluate_test.py` on all 200 images in that folder.

Validation logs use `steps_per_epoch=5`. That run had 25 training batches, 800 training images, and 200 validation images at batch size 32. Each epoch therefore trains on 160 of 800 images and scores 192 validation images (`200 // 32`). The validation generator also shuffles.

### Validation, from the training logs

| Stage | Checkpoint | Epoch | Accuracy | Val accuracy | Val loss | Where |
| --- | --- | --- | --- | --- | --- | --- |
| Feature extraction | `O_R_tlearn_vgg16.keras` | 10/10 | 0.8438 | 0.8177 | 0.3754 | Feature-extraction run. Lowest `val_loss` in that run. |
| Fine-tuning | `O_R_tlearn_fine_tune_vgg16.keras` | 10/10 | 0.9062 | 0.9062 | 0.2629 | Fine-tune run. Highest `val_accuracy` and lowest `val_loss` in that run. |

Earlier fine-tune epochs in that run include validation accuracy 0.8750 (epoch 3), 0.8802 (epoch 7), and 0.8750 (epoch 8). The final logged validation accuracy is 0.9062.

Both stages log `learning_rate: 1.0000e-04` on epoch 1. A `LearningRateScheduler` is attached in both fits. The feature-extraction compile step sets Adam at `1e-5`, and the fine-tune compile step sets RMSprop at `1e-4`. The logged schedule starts at `1e-4` for both.

### Training-run test subset

The saved checkpoints were scored on 100 images: the first 50 paths in `o-vs-r-split/test/O` and the first 50 in `o-vs-r-split/test/R`.

| Model | Accuracy | O precision | O recall | O f1 | R precision | R recall | R f1 | Support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Extract Features Model | 0.77 | 0.75 | 0.80 | 0.78 | 0.79 | 0.74 | 0.76 | 50 + 50 |
| Fine-Tuned Model | 0.79 | 0.78 | 0.80 | 0.79 | 0.80 | 0.78 | 0.79 | 50 + 50 |

Macro and weighted averages in that output match the accuracy for each model (0.77 and 0.79). That output is the classification report only.

### Released model on `data/test`

`scripts/evaluate_test.py --download` scored both `v1.0-models` checkpoints on every image in `data/test` (100 organic, 100 recyclable). Those images are not in git. Figures below are the script's classification report at `digits=4`. Confusion-matrix rows are true labels and columns are predictions, in order O, R.

| Model | Accuracy | O precision | O recall | O f1 | R precision | R recall | R f1 | Support | Confusion matrix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `O_R_tlearn_vgg16.keras` | 0.8400 | 0.8400 | 0.8400 | 0.8400 | 0.8400 | 0.8400 | 0.8400 | 100 + 100 | [[84, 16], [16, 84]] |
| `O_R_tlearn_fine_tune_vgg16.keras` | 0.8500 | 0.8646 | 0.8300 | 0.8469 | 0.8365 | 0.8700 | 0.8529 | 100 + 100 | [[83, 17], [13, 87]] |

For `O_R_tlearn_vgg16.keras`, macro and weighted averages are precision 0.8400, recall 0.8400, and f1 0.8400. For `O_R_tlearn_fine_tune_vgg16.keras`, both averages are precision 0.8506, recall 0.8500, and f1 0.8499.

## Status

The methodology report, the `v1.0-models` release, and the `data/test` scores from `scripts/evaluate_test.py` are in place. The 200 test images are not committed.

## License

The code in `scripts/` and `docs/methodology-report.md` are MIT-licensed. See [LICENSE](LICENSE).

Data: Waste Classification Data by Sashaank Sekar (Kaggle [`techsash/waste-classification-data`](https://www.kaggle.com/datasets/techsash/waste-classification-data)), CC BY-SA 4.0, not redistributed here.

Model weights are fine-tuned from VGG16 (Simonyan & Zisserman, University of Oxford VGG, CC BY 4.0). The [`v1.0-models`](https://github.com/dgluchowski/waste-image-classifier/releases/tag/v1.0-models) release weights are provided for educational use; check the dataset terms before reuse.

This project started as IBM Skills Network coursework. No IBM course materials are included in this repository.
