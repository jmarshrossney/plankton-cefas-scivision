from enum import IntEnum
import logging
from pathlib import Path
from typing import TypeAlias
from urllib.request import urlopen
import zipfile

from PIL import Image
import torch
from torchvision.transforms.v2.functional import pil_to_tensor, resize, to_dtype

from .model import load_model

Tensor: TypeAlias = torch.Tensor

ZENODO_URL = "https://zenodo.org/records/6143685/files/images.zip"
IMAGES_DIR = Path(__file__).with_name("benchmark_images")
IMAGES_ZIP = IMAGES_DIR / "images.zip"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)


class Labels(IntEnum):
    """Labels for the three output classes, enumerated"""
    copepod = 0
    detritus = 1
    non_copepod = 2


class BenchmarkDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        files = list(IMAGES_DIR.glob("*.tif"))
        labels = [
            file.stem[::-1].split("_", maxsplit=1)[1][::-1]
            for file in files
        ]
        
        images = []
        for file in files:
            with file.open("rb") as fp:
                image = Image.open(fp)
                rgb = image.convert("RGB")
                images.append(rgb)

        assert len(labels) == len(images)

        self._files = files
        self._labels = labels
        self._images = images


    def len(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> tuple[Tensor, str, Image.Image, Path]:
        file = self._files[idx]
        label = self._labels[idx]
        image = self._images[idx]
        tensor = to_dtype(
            pil_to_tensor(image),  # permutes HWC -> CHW
            torch.float32, scale=True  # rescales [0, 255] -> [0, 1]
        ).unsqueeze(0) # add batch dim
        
        tensor = resize(tensor, size=[256, 256])

        return tensor, label, image, file
        



def load_dataset() -> BenchmarkDataset:

    # TODO: make robust to existence of zip but not folder, say

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

    dataset = load_dataset()
    model = load_model()

    for inputs in dataset:
        tensor, label, _, _ = inputs
        
        # NOTE: do we know that the outputs were softmaxed during training??
        probs = torch.softmax(model(tensor), dim=1)

        pred = int(torch.argmax(probs, dim=1))
        print(pred, Labels(pred).name, label, getattr(Labels, label))


if __name__ == "__main__":
    main()




