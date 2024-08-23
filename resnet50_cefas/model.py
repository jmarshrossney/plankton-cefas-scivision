"""
This module contains a single function that returns a loaded ResNet50.
"""

import pooch
import torch
from torchvision.models.resnet import Bottleneck, ResNet

WEIGHTS_URL = "doi:10.5281/zenodo.6143685/cop-non-detritus-20211215.pth"
WEIGHTS_HASH = "md5:46fd1665c8b966e472152eb695d22ae3"


def load_model(strip_final_layer: bool = False) -> ResNet:
    """
    Constructs an instance of ResNet50 with weights from the CEFAS/Turing project.

    Args:
        strip_final_layer: Replaces the final linear layer with an
            identity transformation. The output size becoms 2048 rather than 3.

    Returns:
        A ResNet with the CEFAS/Turing weights loaded.

    References:
        Weights: https://zenodo.org/records/6143685
        Related report: https://zenodo.org/records/6799166
    """
    weights_path = pooch.retrieve(url=WEIGHTS_URL, known_hash=WEIGHTS_HASH)
    weights = torch.load(
        weights_path, map_location=torch.device("cpu"), weights_only=True
    )

    # Imitates calling torchvision.models.resnet50 with no weights and then changing
    # the size of the output layer to 3. Refer to source code here:
    # https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#resnet50
    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=3,
    )
    model.load_state_dict(weights)
    model.eval()

    if strip_final_layer:
        # TODO: maybe instead add a forward_hook to the final linear layer
        # so that it returns inputs as well as outputs?
        model.fc = torch.nn.Identity()

    return model
