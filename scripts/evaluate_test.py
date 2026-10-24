#!/usr/bin/env python3
"""Score a released waste-classifier model on data/test.

Organic images go in data/test/O. Recyclable images go in data/test/R.
The trained .keras files are not in git. They are the two assets on the
v1.0-models release:

    https://github.com/dgluchowski/waste-image-classifier/releases/download/v1.0-models/O_R_tlearn_vgg16.keras
    https://github.com/dgluchowski/waste-image-classifier/releases/download/v1.0-models/O_R_tlearn_fine_tune_vgg16.keras

Download one with gh and score it:

    python scripts/evaluate_test.py --download O_R_tlearn_vgg16.keras
    python scripts/evaluate_test.py --download O_R_tlearn_fine_tune_vgg16.keras

Or fetch it yourself, then pass the local path:

    gh release download v1.0-models --repo dgluchowski/waste-image-classifier --pattern 'O_R_tlearn_vgg16.keras'
    python scripts/evaluate_test.py --model O_R_tlearn_vgg16.keras

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
import shutil
import subprocess
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
CLASS_NAMES = ("O", "R")
RELEASE_TAG = "v1.0-models"
RELEASE_REPO = "dgluchowski/waste-image-classifier"
RELEASE_BASE = (
    "https://github.com/dgluchowski/waste-image-classifier/releases/download/v1.0-models"
)
# Byte sizes of the two v1.0-models assets.
RELEASE_ASSETS = {
    "O_R_tlearn_vgg16.keras": 112455527,
    "O_R_tlearn_fine_tune_vgg16.keras": 104061755,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to a .keras file already downloaded from the v1.0-models release",
    )
    parser.add_argument(
        "--download",
        choices=tuple(RELEASE_ASSETS),
        help=(
            "Download this v1.0-models asset into models/ with "
            "`gh release download`, then score it"
        ),
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


def asset_url(name: str) -> str:
    return f"{RELEASE_BASE}/{name}"


def print_release_help(stream) -> None:
    print(
        "The checkpoints are the v1.0-models release assets:",
        file=stream,
    )
    for name in RELEASE_ASSETS:
        print(f"  {asset_url(name)}", file=stream)
    print(
        "Download with: python scripts/evaluate_test.py --download "
        "<asset-name>",
        file=stream,
    )
    print(
        "or: gh release download v1.0-models "
        f"--repo {RELEASE_REPO} --pattern '<asset-name>'",
        file=stream,
    )


def download_asset(name: str, dest: Path) -> bool:
    """Fetch one v1.0-models asset into dest. Returns True when the file matches."""
    expected = RELEASE_ASSETS[name]
    url = asset_url(name)
    if dest.is_file() and dest.stat().st_size == expected:
        print(f"Using {dest} ({expected} bytes) from {url}", flush=True)
        return True

    print(f"Downloading {url}", flush=True)
    gh = shutil.which("gh")
    if gh is None:
        print("gh is not installed, so the asset was not downloaded.", file=sys.stderr)
        print_release_help(sys.stderr)
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            gh,
            "release",
            "download",
            RELEASE_TAG,
            "--repo",
            RELEASE_REPO,
            "--pattern",
            name,
            "--dir",
            str(dest.parent),
            "--clobber",
        ],
        check=False,
    )
    if completed.returncode != 0 or not dest.is_file():
        print(
            f"gh release download failed for {url} (exit {completed.returncode}).",
            file=sys.stderr,
        )
        print_release_help(sys.stderr)
        return False
    actual = dest.stat().st_size
    if actual != expected:
        print(
            f"Downloaded {dest} is {actual} bytes; expected {expected} for {url}.",
            file=sys.stderr,
        )
        return False
    print(f"Downloaded {dest} ({actual} bytes)", flush=True)
    return True


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
    if args.download and args.model is not None:
        print("Pass either --download or --model, not both.", file=sys.stderr)
        return 2
    if args.download is None and args.model is None:
        print("Pass --download NAME or --model PATH.", file=sys.stderr)
        print_release_help(sys.stderr)
        return 2

    if args.download:
        model_path = Path("models") / args.download
        if not download_asset(args.download, model_path):
            print("data/test metrics remain TBD until that file is scored.", file=sys.stderr)
            return 2
    else:
        model_path = args.model

    if not model_path.is_file():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        print_release_help(sys.stderr)
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

    model = tf.keras.models.load_model(model_path)

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

    print(f"Model: {model_path}")
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
    axis.set_title(model_path.name)
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
