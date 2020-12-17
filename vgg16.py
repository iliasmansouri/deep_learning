import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn.functional as F

from utils import DataSplit, conv_layer, fc_layer


class VGG16(pl.LightningModule):
    def __init__(self, path_to_data):
        super(VGG16, self).__init__()

        self.path_to_data = path_to_data
        self.base_net = nn.Sequential(
            conv_layer(3, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            conv_layer(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            conv_layer(128, 256, 3, 1, 1),
            conv_layer(256, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            conv_layer(256, 512, 3, 1, 1),
            conv_layer(512, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            conv_layer(512, 512, 3, 1, 1),
            conv_layer(512, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),
        )
        self.head = nn.Sequential(
            fc_layer(512 * 7 * 7, 4096),
            fc_layer(4096, 4096),
            fc_layer(4096, 3),
            nn.Softmax(),
        )

    def forward(self, x):
        out = self.base_net(x)
        out = out.view(-1, 512 * 7 * 7)
        return self.head(out)

    def prepare_data(self):
        dataset = ImageFolder(
            self.path_to_data,
            transform=transforms.Compose(
                [
                    transforms.CenterCrop(227),
                    transforms.ToTensor(),
                ]
            ),
        )

        data_split = DataSplit(dataset, shuffle=True)
        self.train_loader, self.val_loader, self.test_loader = data_split.get_split(
            batch_size=10, num_workers=8
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def train_dataloader(self):
        return self.train_loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


if __name__ == "__main__":
    path_to_data = "/media/ilias/cc9b802b-6748-4b71-b805-acbbf89c8fb04/home/ilias/Projects/ImageNet-Datasets-Downloader/data/imagenet_images"
    model = VGG16(path_to_data)
    model.prepare_data()
    trainer = pl.Trainer(gpus=1)
    trainer.fit(model)
