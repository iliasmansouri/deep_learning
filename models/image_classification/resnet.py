import torch
import torch.nn as nn
import torch.nn.functional as F
from models.image_classification.abstract import ImageClassifier
from models.nn_utils import conv_layer, fc_layer


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


class ResNet(ImageClassifier):
    def __init__(self, data_handler):
        self.loss_function = F.cross_entropy
        self.optimizer = torch.optim.Adam

        ImageClassifier.__init__(self, data_handler, self.loss_function, self.optimizer)

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
