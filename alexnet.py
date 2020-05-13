import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from utils import DataSplit


class AlexNet(pl.LightningModule):
    def __init__(self, path_to_data):
        super(AlexNet, self).__init__()

        self.path_to_data = path_to_data

        self.base_net = nn.Sequential(
            self.conv_layer_with_LRN(3, 96, 11, 4),
            self.conv_layer_with_LRN(96, 256, 5, padding=2),
            self.conv_layer(256, 384, 3, padding=1),
            self.conv_layer(384, 384, 3, padding=1),
            self.conv_layer(384, 256, 3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.head = nn.Sequential(
            self.fc_layer(256*6*6, 4096),
            self.fc_layer(4096, 4096),
            self.fc_layer(4096, 3)
        )

    def fc_layer(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU())

    def conv_layer(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU())

    def conv_layer_with_LRN(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

    def forward(self, x):
        out = self.base_net(x)
        out = out.view(-1, 256 * 6 * 6)
        return self.head(out)

    def prepare_data(self):
        dataset = ImageFolder(self.path_to_data, transform=transforms.Compose([
            transforms.CenterCrop(227),
            transforms.ToTensor(),
        ]))

        data_split = DataSplit(dataset, shuffle=True)
        self.train_loader, self.val_loader, self.test_loader = data_split.get_split(
            batch_size=128, num_workers=8)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def train_dataloader(self):
        return self.train_loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


if __name__ == "__main__":
    path_to_data = "/mnt/cc9b802b-6748-4b71-b805-acbbf89c8fb0/home/ilias/Projects/data/imagenet_images"
    model = AlexNet(path_to_data)
    model.prepare_data()
    trainer = pl.Trainer(gpus=1)
    trainer.fit(model)
