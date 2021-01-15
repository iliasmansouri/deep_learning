import logging

import torch.nn as nn


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print("-------------here ", x.shape)
        return x


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # dynamically add padding based on kernel_size
        self.padding = (
            self.kernel_size[0] // 2,
            self.kernel_size[1] // 2,
        )


def activation_func(activation):
    return nn.ModuleDict(
        [
            ["relu", nn.ReLU(inplace=True)],
            ["leaky_relu", nn.LeakyReLU(negative_slope=0.01, inplace=True)],
            ["selu", nn.SELU(inplace=True)],
            ["none", nn.Identity()],
        ]
    )[activation]


def fc_layer(in_features, out_features):
    return nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU())


# def conv_layer(in_channels, out_channels, kernel_size, stride=1, padding=0):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), nn.ReLU()
#     )


# def conv_batch_relu_layer(in_channels, out_channels, kernel_size, stride=1, padding=0):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(),
#     )


# def conv_batch_layer(in_channels, out_channels, kernel_size, stride=1, padding=0):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
#         nn.BatchNorm2d(out_channels),
#     )


def conv_layer(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    auto_pad=False,
    use_batchnorm=False,
    activation="relu",
):
    layer = []

    if auto_pad:
        layer.append(
            Conv2dAuto(in_channels, out_channels, kernel_size, stride, padding)
        )
    else:
        layer.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))

    if use_batchnorm:
        layer.append(nn.BatchNorm2d(out_channels))

    layer.append(activation_func(activation))

    return nn.Sequential(*layer)
