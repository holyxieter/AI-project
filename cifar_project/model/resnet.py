import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channel))

    def forward(self, x):
        out1 = self.layer(x)
        out2 = self.shortcut(x)
        out = out1 + out2
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channel = 32

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.layer1 = self.make_layer(ResBlock, 64, 2, 2)
        self.layer2 = self.make_layer(ResBlock, 128, 2, 2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, 2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, 2)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channel, stride, num_block):
        layers_list = []
        for i in range(num_block):
            if i == 0:
                in_stride = stride
            else:
                in_stride = 1
            layers_list.append(block(self.in_channel, out_channel, in_stride))
            self.in_channel = out_channel
        return nn.Sequential(*layers_list)

    def forward(self, x):
        out = self.conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.max_pool2d(out, kernel_size=2)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


def resnet():
    return ResNet(ResBlock)
