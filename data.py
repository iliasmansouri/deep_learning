import logging
import os
from functools import lru_cache

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder


class CocoData:
    def __init__(
        self, coco_path, transform_fn, batch_size=4, shuffle=True, num_workers=8
    ):
        self.coco_path = coco_path
        self.train_path = os.path.join(self.coco_path, "train2017")
        self.test_path = os.path.join(self.coco_path, "test2017")
        self.validation_path = os.path.join(self.coco_path, "val2017")
        self.annotation_path = os.path.join(
            self.coco_path,
            "annotations_trainval2017/annotations/captions_train2017.json",
        )

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.transform = transform_fn

    def get_train_loader(self):
        return DataLoader(
            datasets.CocoDetection(
                root=self.train_path,
                annFile=self.annotation_path,
                transform=self.transform,
            ),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def get_test_loader(self):
        return DataLoader(
            datasets.CocoDetection(
                root=self.test_path,
                annFile=self.annotation_path,
                transform=self.transform,
            ),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def get_validation_loader(self):
        return DataLoader(
            datasets.CocoDetection(
                root=self.validation_path,
                annFile=self.annotation_path,
                transform=self.transform,
            ),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def get_split(self):
        return (
            self.get_train_loader(),
            self.get_validation_loader(),
            self.get_test_loader(),
        )


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
        self.train_loader = DataLoader(
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
        self.val_loader = DataLoader(
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
        self.test_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=self.test_sampler,
            shuffle=False,
            num_workers=num_workers,
        )
        return self.test_loader


class DataHandler:
    def __init__(
        self,
        path,
        data_type,
        augmentation=None,
        batch_size=4,
        shuffle=True,
        num_workers=8,
    ):
        self.path = path
        self.data_type = data_type
        self.transform = transforms.Compose(
            [transforms.CenterCrop(227), transforms.ToTensor()]
        )
        self.augmentation = augmentation

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.dataset = self.get_dataset()

    def get_split(self):
        if self.data_type == "coco":
            return self.dataset.get_split()
        elif self.data_type == "image_folder":
            return DataSplit(self.dataset, shuffle=True).get_split(
                batch_size=128, num_workers=8
            )

    def get_dataset(self):
        if self.data_type == "coco":
            return CocoData(
                self.path,
                self.transform,
                self.batch_size,
                self.shuffle,
                self.num_workers,
            )
        elif self.data_type == "image_folder":
            return ImageFolder(self.path, self.transform)
        else:
            raise ValueError(
                "Data type can be either coco or image_folder but is: %s",
                self.data_type,
            )

    def get_num_classes(self):
        if self.data_type == "coco":
            return 80
        if self.data_type == "image_folder":
            return 3
