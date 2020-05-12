import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(33 * 33 * 96, 3)

    def forward(self, x):
        out = self.layer1(x)
        out = self.fc1(x)
        return out


if __name__ == "__main__":
    path_to_data = "data/imagenet_images"
    dataset = CustomDataset(path_to_data)
    