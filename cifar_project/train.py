import torch
import torch.nn as nn
import torchvision
from model.alexnet import AlexNet
from model.googlenet import GoogLeNet
from model.lenet import LeNet
from model.resnet import ResNet
from model.vggnet import VGGNet
from load_data import train_loader, test_loader
import matplotlib.pyplot as plt


def train_model(model, train_data, val_data, criterion, optimizer, scheduler, num_epochs, device):
    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_correct = 0
        total_samples = 0
        running_loss = 0.0

        for i, data in enumerate(train_data):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, dim=1)
            total_correct += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_accuracy = 100.0 * (total_correct.double() / total_samples)

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        model.eval()
        val_total_correct = 0
        val_total_samples = 0
        val_running_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(val_data):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, dim=1)
                val_total_correct += torch.sum(preds == labels.data)
                val_total_samples += inputs.size(0)

        val_epoch_loss = val_running_loss / val_total_samples
        val_epoch_accuracy = 100.0 * (val_total_correct.double() / val_total_samples)

        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)

        print(f'Epoch {epoch}/{num_epochs - 1}, '
              f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%, '
              f'Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_accuracy:.2f}%')

        scheduler.step()

    return (torch.tensor(train_accuracies).to(device).cpu().numpy().tolist(),
            torch.tensor(train_losses).to(device).cpu().numpy().tolist(),
            torch.tensor(val_accuracies).to(device).cpu().numpy().tolist(),
            torch.tensor(val_losses).to(device).cpu().numpy().tolist())


if __name__ == '__main__':
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_num = 100
    lr = 0.001

    net = LeNet().to(device)
    train_data = list(train_loader)
    val_data = list(test_loader)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-3)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    train_accuracies, train_losses, val_accuracies, val_losses = train_model(
        net, train_data, val_data, criterion, optimizer, scheduler, epoch_num, device)

    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.plot(train_accuracies, label='Training Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(train_losses, label='Training Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(val_accuracies, label='Validation Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
