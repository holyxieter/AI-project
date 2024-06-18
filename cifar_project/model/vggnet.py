import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()

        # 32 * 32 * 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.max_pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 16 * 16 * 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.max_pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 8 * 8 * 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.max_pooling3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 4 * 4 * 256
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.max_pooling4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2 * 2 * 512
        self.fc1 = nn.Linear(2 * 2 * 512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        out = self.conv1(x)
        out = self.max_pooling1(out)

        out = self.conv2(out)
        out = self.max_pooling2(out)

        out = self.conv3(out)
        out = self.max_pooling3(out)

        out = self.conv4(out)
        out = self.max_pooling4(out)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out


def vggnet():
    return VGGNet()
