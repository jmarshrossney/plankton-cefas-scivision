from typing import TypeAlias

from torch.utils.data import Dataset
import torchvision
import xarray
import numpy
import torch

# Could be Dataset of pandas DataFrame as far as I can see
InputDataType: TypeAlias = xarray.Dataset


class PlanktonDataset(Dataset):
    """
    A image-based dataset.

    Indexing the dataset using an integer key returns a tuple containing
    (a) the raster, and (b) a torch.Tensor that has been resized to
    the standard (256, 256, X).

    Args:
        An xarray.Dataset containing images with X format and Y fields
    """
    def __init__(self, ds: InputDataType):
        self.ds = ds
        self.n_images = self.ds.dims['concat_dim']
        self.img_ixs = self.ds.concat_dim.values

    def __len__(self) -> int:
        return self.n_images

    def __getitem__(self, idx: int) -> tuple[numpy.ndarray, torch.Tensor]:
        im_raw = self.ds.sel(concat_dim=self.img_ixs[idx])
        imw = im_raw.image_width.values
        iml = im_raw.image_length.values

        im_ori = im_raw['raster'][0:iml, 0:imw, :].values

        im_model = torchvision.transforms.ToTensor()(im_ori)
        im_model = torchvision.transforms.Resize((256,256))(im_model)

        X_raw = im_raw['raster'].values #raw image
        X_model = im_model #image transformed for prediction

        return X_raw, X_model
