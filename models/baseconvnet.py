# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Any


class ConvNetSimple(nn.Module):
    def __init__(self, channels=6):
        super().__init__()
        self.model = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(128, 1, kernel_size=1, padding=0),
                nn.Sigmoid())
    
    def forward(self, x):
        return self.model(x)