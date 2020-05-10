"""
Neural Network models for training and testing implemented in PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from deep_noise_rejection.MNIST.models.tools import Normalize


class CNN(nn.Module):

    # 2 Conv layers, 2 Fc layers

    def __init__(self):
        super(CNN, self).__init__()

        self.norm = Normalize(mean=[0.1307], std=[0.3081])
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5,
                               stride=1, padding=2, bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5,
                               stride=1, padding=2, bias=True)
        self.fc1 = nn.Linear(28 * 28 * 64, 1024, bias=True)
        self.fc2 = nn.Linear(1024, 10, bias=True)

    def forward(self, x):

        x = self.norm(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class CNN_standard(nn.Module):

    # 2 Conv layers, 2 Fc layers

    def __init__(self):
        super(CNN_standard, self).__init__()

        self.norm = Normalize(mean=[0.1307], std=[0.3081])
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5,
                               stride=1, padding=2, bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5,
                               stride=1, padding=2, bias=True)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024, bias=True)
        self.fc2 = nn.Linear(1024, 10, bias=True)

    def forward(self, x):

        x = self.norm(x)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
