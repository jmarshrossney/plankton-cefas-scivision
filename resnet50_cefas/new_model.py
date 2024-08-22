import pooch
import torch
from torchvision.models.resnet import Bottleneck, ResNet

WEIGHTS_URL = "doi:10.5281/zenodo.6143685/cop-non-detritus-20211215.pth"
WEIGHTS_HASH = "md5:46fd1665c8b966e472152eb695d22ae3"

def build_model():
    weights_path = pooch.retrieve(url=WEIGHTS_URL, known_hash=WEIGHTS_HASH)
    weights = torch.load(weights_path, map_location=torch.device("cpu"), weights_only=True)

    # Imitates calling resnet50 -> _resnet with no weights and then changing
    # the size of the output layer to 3
    model = ResNet(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            num_classes=3,
    )
    model.load_state_dict(weights)
    model.eval()

    return model


if __name__ == "__main__":
    build_model()

