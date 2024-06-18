import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool):
        super(Inception, self).__init__()

        # 1x1 conv branch
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)

        # 1x1 conv -> 3x3 conv branch
        self.branch3x3_1 = nn.Conv2d(in_channels, red_3x3, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1)

        # 1x1 conv -> 5x5 conv branch
        self.branch5x5_1 = nn.Conv2d(in_channels, red_5x5, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2)

        # 3x3 pool -> 1x1 conv branch
        self.branch_pool = nn.Conv2d(in_channels, out_pool, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()

        self.pre_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.a3 = Inception(32, 16, 16, 16, 4, 8, 8)
        self.b3 = Inception(48, 24, 24, 24, 6, 12, 12)

        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)

        self.a4 = Inception(72, 32, 32, 32, 8, 16, 16)
        self.b4 = Inception(96, 48, 48, 48, 12, 24, 24)

        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(17424, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.dropout(self.fc1(out))
        out = self.dropout(self.fc2(out))
        out = self.fc3(out)
        return out


def googlenet():
    return GoogLeNet()
