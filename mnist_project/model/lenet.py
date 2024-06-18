import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)  # Output: 28x28x6
        self.s2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 14x14x6
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)  # Output: 10x10x16
        self.s4 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 5x5x16
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)  # Output: 1x1x120
        self.ReLU = nn.ReLU()
        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

    def forward(self, x):
        x = self.ReLU(self.c1(x))
        x = self.s2(x)
        x = self.ReLU(self.c3(x))
        x = self.s4(x)
        x = self.ReLU(self.c5(x))
        x = self.flatten(x)
        x = self.ReLU(self.f6(x))
        x = self.output(x)
        return x


def lenet():
    return LeNet()
