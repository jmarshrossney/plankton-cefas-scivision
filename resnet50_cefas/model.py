import numpy as np
import torch
import torchvision
from .data import PlanktonDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from .utils import PlanktonLabels
import pooch
import random

class resnet50:
    def __init__(self, model_weights: dict = None, label_level=None):

        if label_level is None:
            label_level = 'label2_detritus'

        if model_weights is None:
            model_weights = dict(url="doi:10.5281/zenodo.6143685/cop-non-detritus-20211215.pth",
                             known_hash="md5:46fd1665c8b966e472152eb695d22ae3")

        # ---- DOWNLOAD
        self.model_weights = pooch.retrieve(url=model_weights['url'], known_hash=model_weights['known_hash'])

        # ---- LABEL LEVEL
        self.labels_map = PlanktonLabels(experiment=label_level).labels()
        target_classes = len(self.labels_map.values())

        # ---- LOAD PRETRAINED MODEL
        model = torchvision.models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, target_classes)

        # replace default weights by the fine-tune model
        model.load_state_dict(torch.load(self.model_weights, map_location=torch.device('cpu'), weights_only=True))  # path of your weights

        #initialise the model in evaluation mode
        self.pretrained_model = model
        self.pretrained_model.eval()

    def get_sample(self, ds, idx):
        im_target = ds.sel(concat_dim=idx)
        image = im_target['raster'].values
        iml = im_target.image_length.values
        imw = im_target.image_width.values
        return idx, image[0:iml, 0:imw, :]

    def show_output(self, obs, preds, xarray=False):

        if xarray:
            plt.figure(figsize=(20, 20))
            columns = 5
            samples = [self.get_sample(obs, idx=i) for i in range(26)]
            for i, image in enumerate(samples):
                plt.subplot(int(len(samples) / columns + 1), columns, i + 1)
                plt.imshow(image[1])
                plt.axis('off')
                pred_target = preds[image[0]]
                plt.title("Prediction: {}".format(self.labels_map[int(pred_target.detach().numpy())]), fontsize='xx-large')
        else:
            plt.figure()
            plt.imshow(obs)
            plt.title("Prediction: {}".format(self.labels_map[int(preds.detach().numpy())]), fontsize='xx-large')
            plt.axis('off')
        plt.show()

    def predict_batch(self, image: np.ndarray, batch_size: int):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # add with and length info
        image = image.assign(
            image_width = image['EXIF Image ImageWidth'].to_pandas().apply(lambda x: x.values[0]),
            image_length = image['EXIF Image ImageLength'].to_pandas().apply(lambda x: x.values[0])
        )

        dataset = PlanktonDataset(image)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_labels = []
        with torch.no_grad():
            for i, (_, inputs) in enumerate(dataloader):
                inputs = inputs.to(device)
                outputs = self.pretrained_model(inputs)
                _, preds = torch.max(outputs, 1)
                all_labels.append(preds)

        all_labels_flatten = torch.cat(all_labels)

        self.show_output(image,all_labels_flatten,xarray=True)

        return all_labels_flatten

    def predict(self, image: np.ndarray) -> np.ndarray:

        X = torchvision.transforms.ToTensor()(image)
        X = torchvision.transforms.Resize((256, 256))(X)
        X = torch.unsqueeze(X, 0)

        y = self.pretrained_model(X)

        _, preds = torch.max(y, 1)

        self.show_output(image,preds,xarray=False)

        return preds


if __name__ == "__main__":
    pass
