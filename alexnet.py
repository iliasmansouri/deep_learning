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

        self.layer1 = self.conv_layer_with_LRN(3, 96, 11, 4)
        self.layer2 = self.conv_layer_with_LRN(96, 256, 5, padding=2)

        self.layer3 = self.conv_layer(256, 384, 3, padding=1)
        self.layer4 = self.conv_layer(384, 384, 3, padding=1)

        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.base_net = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )

        self.fc1 = self.fc_layer(256*6*6, 4096)
        self.fc2 = self.fc_layer(4096, 4096)
        self.fc3 = self.fc_layer(4096, 3)

        self.head = nn.Sequential(
            self.fc1,
            self.fc2,
            self.fc3
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


if __name__ == "__main__":
    path_to_data = "/mnt/cc9b802b-6748-4b71-b805-acbbf89c8fb0/home/ilias/Projects/data/imagenet_images"

    dataset = ImageFolder(path_to_data, transform=transforms.Compose([
        transforms.CenterCrop(227),
        transforms.ToTensor(),
    ]))

    dataloader = DataLoader(
        dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        batch_size=128)

    model = AlexNet()
    print(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())

    epochs = 10
    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        accuracy = 0

        # Training the model
        model.train()
        counter = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*inputs.size(0)

            counter += 1
            print(counter, "/", len(dataloader))
