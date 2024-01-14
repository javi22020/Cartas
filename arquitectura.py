import torch
from torch import nn
from torch.nn import functional as F

class Arquitectura(nn.Module):
    def __init__(self, numclasses: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Capas convolucionales
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        # Capas de pooling
        self.pool = nn.MaxPool2d(3)
        # Capas de dropout
        self.dropout = nn.Dropout(0.25)
        # Funciones de activacion
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # Capas de flatten
        self.flatten = nn.Flatten()
        # Capas de batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        # Capas lineales
        self.lin1 = nn.Linear(256, 4096)
        self.lin2 = nn.Linear(4096, 256)
        self.lin3 = nn.Linear(256, numclasses)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.lin1(x)
        x = self.sigmoid(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.sigmoid(x)
        x = self.dropout(x)
        x = self.lin3(x)
        return x