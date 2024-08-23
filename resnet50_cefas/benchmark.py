"""
This module contains the setup and execution of a benchmark for the CEFAS/Turing model.
"""

import logging
import zipfile
from pathlib import Path
from urllib.request import urlopen

import torch
from PIL import Image
from torchvision.transforms.v2.functional import pil_to_tensor, resize, to_dtype
from torchmetrics import Accuracy, ConfusionMatrix, MetricCollection, Precision, Recall

from .model import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ZENODO_URL = "https://zenodo.org/records/6143685/files/images.zip"
IMAGES_DIR = Path(__file__).with_name("benchmark_images")
IMAGES_ZIP = IMAGES_DIR / "images.zip"

from enum import IntEnum


class Labels(IntEnum):
    """Labels for the three output classes, enumerated"""

    copepod = 0
    detritus = 1
    non_copepod = 2

    @classmethod
    def as_tuple(cls) -> tuple[str, str, str]:
        return tuple(cls(i).name for i in range(3))


class BenchmarkDataset(torch.utils.data.Dataset):
    """
    Wrapper for a directory of `.tif` RGB images.

    The directory is expected to contain the 26 plankton images supplied by
    the original authors and hosted here: https://zenodo.org/records/6143685

    The dataset is accessed by an integer index, which returns a tuple containing
        (a) a representation of the image as a float32 tensor, rescaled to [0, 1],
            and interpolated such that the spatial dimensions are (256, 256)
        (b) a string which is the correct class label
        (c) the original PIL Image (for visualisation)
        (d) the image file path (for sanity checking)

    The labels are inferred from the file names, as in:
        'copepod_X.tif'     -> 'copepod'
        'detritus_X.tif'    -> 'detritus'
        'non_copepod_X.tif' -> 'non_copepod'
    """

    def __init__(self):
        super().__init__()

        files = list(IMAGES_DIR.glob("*.tif"))
        if (n := len(files)) != 26:
            logging.warning("Expected to find 26 images, but found %d" % n)

        labels = [file.stem[::-1].split("_", maxsplit=1)[1][::-1] for file in files]
        assert all([label in Labels.as_tuple() for label in set(labels)])

        labels_tensor = torch.tensor([getattr(Labels, label) for label in labels])

        images = []
        for file in files:
            with file.open("rb") as fp:
                image = Image.open(fp)
                rgb = image.convert("RGB")
                images.append(rgb)

        assert len(labels) == len(images)

        self._files = files
        self._labels = labels
        self._labels_tensor = labels_tensor
        self._images = images

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, Image.Image, Path]:
        file = self._files[idx]
        label = self._labels_tensor[idx]
        image = self._images[idx]

        tensor = to_dtype(
            pil_to_tensor(image),  # permutes HWC -> CHW
            torch.float32,
            scale=True,  # rescales [0, 255] -> [0, 1]
        )

        # NOTE: copied from original, but do we really want to do this?
        tensor = resize(tensor, size=[256, 256])

        return tensor, label, image, file


def load_dataset() -> BenchmarkDataset:
    """
    Wrapper for BenchmarkDataset which downloads and extracts data if required.
    """

    # Download zipfile from Zenodo if it doesn't already exist
    if not IMAGES_DIR.exists():
        logging.info("Creating directory at %s" % IMAGES_DIR)
        IMAGES_DIR.mkdir(parents=True, exist_ok=False)

        logging.info("Downloading archive from %s" % ZENODO_URL)
        with urlopen(ZENODO_URL) as response:
            with IMAGES_ZIP.open("wb") as zf:
                zf.write(response.read())

        logging.info("Extracting contents of %s" % IMAGES_ZIP)
        with zipfile.ZipFile(IMAGES_ZIP, "r") as zip_ref:
            zip_ref.extractall(IMAGES_DIR)

    else:
        logging.info("Using existing images in %s" % IMAGES_DIR)

    return BenchmarkDataset()


def main():
    """
    Benchmarks the CEFAS/Turing model on the 26-image dataset.

    Computes the accuracy and confusion matrix, prints each of these to stdout,
    and saves an image of the confusion matrix to `confusion_matrix.png`.

    Currently this expects the images to have compatible dimensions so that
    the forward pass can be batched.
    """
    dataset = load_dataset()
    model = load_model()

    inputs, targets, _, _ = list(zip(*dataset))
    # NOTE: this only works when the dimensions are standardised
    inputs, targets = map(torch.stack, [inputs, targets])

    outputs = model(inputs)
    preds = torch.softmax(outputs, dim=1)

    acc = Accuracy(task="multiclass", num_classes=3)
    print(f"Accuracy: {float(acc(preds, targets)):.2f}")

    cm = ConfusionMatrix(task="multiclass", num_classes=3)
    print(f"Confusion matrix: {cm(preds, targets)}")

    logger.info("Saving confusion matrix to confusion_matrix.png")
    fig, _ = cm.plot(labels=Labels.as_tuple())
    fig.savefig("confusion_matrix.png")


if __name__ == "__main__":
    main()
