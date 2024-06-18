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
        return torch.cat(outputs, dim=1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()

        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.a3 = Inception(32, 32, 48, 64, 8, 16, 16)  # 32 + 64 + 16 + 16 = 128
        self.b3 = Inception(128, 64, 64, 96, 16, 48, 32)  # 64 + 96 + 48 + 32 = 240

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(240, 96, 48, 104, 8, 24, 32)  # 96 + 104 + 24 + 32 = 256
        self.b4 = Inception(256, 80, 56, 112, 12, 32, 32)  # 80 + 112 + 32 + 32 = 256
        self.c4 = Inception(256, 64, 64, 128, 12, 32, 32)  # 64 + 128 + 32 + 32 = 256
        self.d4 = Inception(256, 56, 72, 144, 16, 32, 32)  # 56 + 144 + 32 + 32 = 264
        self.e4 = Inception(264, 128, 80, 160, 16, 64, 64)  # 128 + 160 + 64 + 64 = 416

        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.a5 = Inception(416, 128, 80, 160, 16, 64, 64)  # 128 + 160 + 64 + 64 = 416
        self.b5 = Inception(416, 192, 96, 192, 24, 64, 64)  # 192 + 192 + 64 + 64 = 512

        self.avgpool = nn.AvgPool2d(8)
        self.dropout = nn.Dropout(0.4)
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 10)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool2(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.dropout(self.linear1(out))
        out = self.dropout(self.linear2(out))
        out = self.linear3(out)
        return out


def googlenet():
    return GoogLeNet()
