import torch.nn as nn
import torch.nn.functional as F


class Raw_CNN(nn.Module):
    def __init__(self):
        super(Raw_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, 5, padding=1, bias=True)
        self.conv2 = nn.Conv2d(12, 64, 5, padding=1, bias=True)
        self.m1 = nn.MaxPool2d(2)
        self.m2 = nn.MaxPool2d(2, padding=1)
        self.fc1 = nn.Linear(64 * 6 * 6, 10, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (3, 24, 24) -> (64, 20, 20)
        x = self.m1(x)  # (64, 10, 10)
        x = F.relu(self.conv2(x))  # (64, 6, 6)
        x = self.m2(x)  # (64, 3, 3)
        x = x.view(-1, 64 * 6 * 6)  # (64*3*3)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class Tailored_CNN(nn.Module):
    def __init__(self):
        super(Tailored_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, 5, padding=1, bias=False)
        self.conv2 = nn.Conv2d(12, 64, 5, padding=1, bias=False)
        self.a1 = nn.AvgPool2d(2)
        self.a2 = nn.AvgPool2d(2, padding=1)
        self.fc1 = nn.Linear(64 * 6 * 6, 10, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (3, 24, 24) -> (64, 20, 20)
        x = self.a1(x)  # (64, 10, 10)
        x = F.relu(self.conv2(x))  # (64, 6, 6)
        x = self.a2(x)  # (64, 3, 3)
        x = x.view(-1, 64 * 6 * 6)  # (64*3*3)
        x = self.fc1(x)  # (10*1*1)
        return F.log_softmax(x, dim=1)
