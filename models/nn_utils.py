import logging
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler


class DataSplit:
    def __init__(
        self, dataset, test_train_split=0.8, val_train_split=0.1, shuffle=False
    ):
        self.dataset = dataset

        dataset_size = len(dataset)
        self.indices = list(range(dataset_size))
        test_split = int(np.floor(test_train_split * dataset_size))

        if shuffle:
            np.random.shuffle(self.indices)

        train_indices, self.test_indices = (
            self.indices[:test_split],
            self.indices[test_split:],
        )
        train_size = len(train_indices)
        validation_split = int(np.floor((1 - val_train_split) * train_size))

        self.train_indices, self.val_indices = (
            train_indices[:validation_split],
            train_indices[validation_split:],
        )

        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.val_sampler = SubsetRandomSampler(self.val_indices)
        self.test_sampler = SubsetRandomSampler(self.test_indices)

    def get_train_split_point(self):
        return len(self.train_sampler) + len(self.val_indices)

    def get_validation_split_point(self):
        return len(self.train_sampler)

    @lru_cache(maxsize=4)
    def get_split(self, batch_size=50, num_workers=4):
        logging.debug("Initializing train-validation-test dataloaders")
        self.train_loader = self.get_train_loader(
            batch_size=batch_size, num_workers=num_workers
        )
        self.val_loader = self.get_validation_loader(
            batch_size=batch_size, num_workers=num_workers
        )
        self.test_loader = self.get_test_loader(
            batch_size=batch_size, num_workers=num_workers
        )
        return self.train_loader, self.val_loader, self.test_loader

    @lru_cache(maxsize=4)
    def get_train_loader(self, batch_size=50, num_workers=4):
        logging.debug("Initializing train dataloader")
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=self.train_sampler,
            shuffle=False,
            num_workers=num_workers,
        )
        return self.train_loader

    @lru_cache(maxsize=4)
    def get_validation_loader(self, batch_size=50, num_workers=4):
        logging.debug("Initializing validation dataloader")
        self.val_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=self.val_sampler,
            shuffle=False,
            num_workers=num_workers,
        )
        return self.val_loader

    @lru_cache(maxsize=4)
    def get_test_loader(self, batch_size=50, num_workers=4):
        logging.debug("Initializing test dataloader")
        self.test_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=self.test_sampler,
            shuffle=False,
            num_workers=num_workers,
        )
        return self.test_loader


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
