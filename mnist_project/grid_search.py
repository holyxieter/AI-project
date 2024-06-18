import torch
import torch.nn as nn
import numpy as np
from model.alexnet import AlexNet
from model.googlenet import GoogLeNet
from model.lenet import LeNet
from model.resnet import ResNet
from model.vggnet import VGGNet
from load_data import train_loader, test_loader


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

        scheduler.step()

    return (torch.tensor(train_accuracies).to(device).cpu().numpy().tolist(),
            torch.tensor(train_losses).to(device).cpu().numpy().tolist(),
            torch.tensor(val_accuracies).to(device).cpu().numpy().tolist(),
            torch.tensor(val_losses).to(device).cpu().numpy().tolist())


if __name__ == '__main__':
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_num = 100

    learning_rates = [0.01, 0.006, 0.001]
    weight_decays = [5e-2, 5e-3, 5e-4]
    gammas = [0.99, 0.95, 0.9]

    best_val_acc = 0.0
    best_params = {}

    train_data = list(train_loader)
    val_data = list(test_loader)
    criterion = nn.CrossEntropyLoss()

    for lr in learning_rates:
        for wd in weight_decays:
            for gamma in gammas:
                net = VGGNet().to(device)
                optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

                print(f'Training with lr={lr}, weight_decay={wd}, gamma={gamma}')
                train_accuracies, train_losses, val_accuracies, val_losses = train_model(
                    net, train_data, val_data, criterion, optimizer, scheduler, epoch_num, device)

                if max(val_accuracies) > best_val_acc:
                    best_val_acc = max(val_accuracies)
                    best_params = {
                        'learning_rate': lr,
                        'weight_decay': wd,
                        'gamma': gamma,
                        'train_accuracies': train_accuracies,
                        'train_losses': train_losses,
                        'val_accuracies': val_accuracies,
                        'val_losses': val_losses
                    }

    print('Best Validation Accuracy: ', best_val_acc)
    print('Best Hyperparameters: ', best_params)
