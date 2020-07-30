import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from utils import conv_batch_relu_layer, fc_layer, DataSplit, conv_batch_layer
from torchvision.datasets import ImageFolder
from torchvision import transforms


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, downsample=None):
        super(ResBlock, self).__init__()

        self.layer1 = conv_batch_relu_layer(
            in_channels, out_channels, kernel_size, stride, padding)
        self.layer2 = conv_batch_layer(
            out_channels, out_channels, kernel_size, stride, padding)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)


class ResNet(pl.LightningModule):
    def __init__(self, path_to_data, num_classes):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.path_to_data = path_to_data
        self.create_conv_layers()

        self.model = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            fc_layer(512*7*7, self.num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        out = self.model(x)
        out

    def create_conv_layers(self):
        self.conv1 = conv_batch_relu_layer(3, 64, 7, 2)
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            conv_batch_relu_layer(64, 64, 3),
            conv_batch_relu_layer(64, 64, 3)
        )
        self.conv3 = nn.Sequential(
            conv_batch_relu_layer(128, 128, 3, 2),
            conv_batch_relu_layer(128, 128, 3)
        )
        self.conv4 = nn.Sequential(
            conv_batch_relu_layer(256, 256, 3, 2),
            conv_batch_relu_layer(256, 256, 3)
        )
        self.conv5 = nn.Sequential(
            conv_batch_relu_layer(512, 512, 3, 2),
            conv_batch_relu_layer(512, 512, 3)
        )


if __name__ == "__main__":
    path_to_data = "/mnt/cc9b802b-6748-4b71-b805-acbbf89c8fb0/home/ilias/Projects/data/imagenet_images"
    model = ResNet(path_to_data, 3)
    model.prepare_data()
    trainer = pl.Trainer(gpus=1)
    trainer.fit(model)
