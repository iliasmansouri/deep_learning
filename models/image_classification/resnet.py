import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.flatten import Flatten
import torch.optim as optim
import pytorch_lightning as pl
from models.nn_utils import (
    fc_layer,
    conv_layer,
)
from torchvision.datasets import ImageFolder
from torchvision import transforms
from data import DataSplit


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResBlock, self).__init__()

        self.layer1 = conv_layer(
            in_channels, out_channels, kernel_size, stride, padding, use_batchnorm=True
        )
        self.layer2 = conv_layer(
            out_channels,
            out_channels,
            kernel_size,
            1,
            padding,
            use_batchnorm=True,
            activation="none",
        )

        self.downsample = conv_layer(
            in_channels, out_channels, kernel_size, 2, use_batchnorm=True, auto_pad=True
        )

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x) if self.stride != 1 else x

        out = self.layer1(x)
        out = self.layer2(out)

        out += residual
        return self.relu(out)


class ResNet(pl.LightningModule):
    def __init__(self, data_handler):
        super(ResNet, self).__init__()
        self.num_classes = data_handler.get_num_classes()
        self.data_handler = data_handler
        self.create_conv_layers()

        self.model = nn.Sequential(
            self.conv1,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            fc_layer(512, self.num_classes),
            nn.Softmax(),
        )

    def forward(self, x):
        out = self.model(x)
        return self.head(out)

    def create_conv_layers(self):
        self.conv1 = conv_layer(3, 64, 7, 2, use_batchnorm=True)
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            ResBlock(64, 64, 3),
            ResBlock(64, 64, 3),
        )
        self.conv3 = nn.Sequential(ResBlock(64, 128, 3, 2), ResBlock(128, 128, 3))
        self.conv4 = nn.Sequential(ResBlock(128, 256, 3, 2), ResBlock(256, 256, 3))
        self.conv5 = nn.Sequential(ResBlock(256, 512, 3, 2), ResBlock(512, 512, 3))

    def prepare_data(self):
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
        ) = self.data_handler.get_split()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        print("results: ", (y, y_hat))
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def train_dataloader(self):
        return self.train_loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
