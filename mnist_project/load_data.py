import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_data_loaders(batch_size=128, num_workers=10, persistent_workers=True):
    # 数据预处理
    data_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # 加载训练集
    train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              persistent_workers=persistent_workers)

    # 加载测试集
    test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             persistent_workers=persistent_workers)

    return train_loader, test_loader
