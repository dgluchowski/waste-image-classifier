#!/usr/bin/env python3
"""Score a released waste-classifier model on data/test.

Organic images go in data/test/O. Recyclable images go in data/test/R.
The trained .keras files are not in git. Download one from the GitHub Release
and pass it with --model:

    python scripts/evaluate_test.py --model /path/to/O_R_tlearn_fine_tune_vgg16.keras

Preprocessing matches the notebook evaluation cell (execution count 29 in
notebooks/waste_classifier_vgg16.ipynb): resize to 150x150, scale pixels to
[0, 1], and threshold the sigmoid output at 0.5. A probability below 0.5 is
organic (O). Otherwise the prediction is recyclable (R), which is class 1 in
the notebook encoding.

This script prints a classification report and writes a confusion matrix.
It does not fill in metrics when the model file or the test images are missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
CLASS_NAMES = ("O", "R")
RELEASE_NAMES = (
    "O_R_tlearn_vgg16.keras",
    "O_R_tlearn_fine_tune_vgg16.keras",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to a released .keras model file",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("data/test"),
        help="Directory that contains O/ and R/ image folders (default: data/test)",
    )
    parser.add_argument("--image-size", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid cutoff. Below this value the label is O (default: 0.5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for confusion_matrix.csv and confusion_matrix.png",
    )
    return parser.parse_args()


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_images(paths: list[Path], image_size: int):
    import numpy as np
    from tensorflow.keras.preprocessing.image import img_to_array, load_img

    images = []
    for path in paths:
        image = load_img(path, target_size=(image_size, image_size))
        array = img_to_array(image).astype("float32")
        array /= 255.0
        images.append(array)
    return np.stack(images, axis=0)


def main() -> int:
    args = parse_args()

    if not args.model.is_file():
        print(f"Model file not found: {args.model}", file=sys.stderr)
        print(
            "Download one of these files from the GitHub Release and pass it with --model:",
            file=sys.stderr,
        )
        for name in RELEASE_NAMES:
            print(f"  {name}", file=sys.stderr)
        print("data/test metrics remain TBD until that file is scored.", file=sys.stderr)
        return 2

    images_by_class = {name: list_images(args.test_dir / name) for name in CLASS_NAMES}
    total = sum(len(paths) for paths in images_by_class.values())
    if total == 0:
        print(
            f"No images found in {args.test_dir}/O or {args.test_dir}/R.",
            file=sys.stderr,
        )
        print(
            "Copy organic images into data/test/O and recyclable images into data/test/R.",
            file=sys.stderr,
        )
        print("data/test metrics remain TBD until those images are scored.", file=sys.stderr)
        return 2

    for name in CLASS_NAMES:
        if not images_by_class[name]:
            print(f"Warning: {args.test_dir / name} has no images.", file=sys.stderr)

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import tensorflow as tf
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError as exc:
        print("Missing dependency. Install with: pip install -r requirements.txt", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1

    model = tf.keras.models.load_model(args.model)

    paths: list[Path] = []
    labels: list[str] = []
    for name in CLASS_NAMES:
        for path in images_by_class[name]:
            paths.append(path)
            labels.append(name)

    probabilities = []
    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start : start + args.batch_size]
        batch = load_images(batch_paths, args.image_size)
        prediction = model.predict(batch, verbose=0).reshape(-1)
        probabilities.append(prediction)
    probabilities = np.concatenate(probabilities, axis=0)
    predicted = np.where(probabilities < args.threshold, "O", "R")

    print(f"Model: {args.model}")
    print(
        "Images: "
        f"{total} (O={len(images_by_class['O'])}, R={len(images_by_class['R'])})"
    )
    print(
        classification_report(
            labels,
            predicted,
            labels=list(CLASS_NAMES),
            digits=4,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(labels, predicted, labels=list(CLASS_NAMES))
    print("Confusion matrix (rows are true labels, columns are predictions; order O, R):")
    print(matrix)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "confusion_matrix.csv"
    png_path = args.output_dir / "confusion_matrix.png"
    np.savetxt(
        csv_path,
        matrix,
        fmt="%d",
        delimiter=",",
        header="pred_O,pred_R",
        comments="",
    )

    figure, axis = plt.subplots(figsize=(4.2, 4.2))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], labels=list(CLASS_NAMES))
    axis.set_yticks([0, 1], labels=list(CLASS_NAMES))
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(args.model.name)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(int(matrix[row, column])), ha="center", va="center")
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(png_path, dpi=150)
    plt.close(figure)

    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
