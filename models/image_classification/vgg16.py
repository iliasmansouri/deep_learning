import torch
import torch.nn as nn
import torch.nn.functional as F
from models.image_classification.abstract import ImageClassifier
from models.nn_utils import conv_layer, fc_layer


class VGG16(ImageClassifier):
    def __init__(self, data_handler):
        self.loss_function = F.cross_entropy
        self.optimizer = torch.optim.Adam

        ImageClassifier.__init__(self, data_handler, self.loss_function, self.optimizer)

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
            fc_layer(4096, self.num_classes),
            nn.Softmax(),
        )

    def forward(self, x):
        out = self.base_net(x)
        out = out.view(-1, 512 * 7 * 7)
        return self.head(out)
