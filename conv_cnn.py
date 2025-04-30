# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 10:19:56 2025

@author: user
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from timeit import default_timer as timer 
from plot import plot_loss
from plot import plot_accuracy
from conf_matrix import conf_mat
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report



class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(1024, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 26)
        

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)  
        x = self.conv7(x)
        x = self.global_pool(x)  
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

def load_data(train_dir, val_dir, test_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    class_names = train_dataset.classes

    return train_loader, val_loader, test_loader, class_names



def train(model, train_loader, loss_fn, optimizer, device):
    model.train()
    train_loss, correct = 0, 0
    total = 0

    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (y_pred.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    
    
    accuracy = 100 * correct / total

    return train_loss / len(train_loader), accuracy  


def validate(model, val_loader, loss_fn, device):
    model.eval()
    val_loss, correct = 0, 0
    total = 0

    with torch.inference_mode():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            val_loss += loss_fn(y_pred, y).item()
            correct += (y_pred.argmax(dim=1) == y).sum().item()
            total += y.size(0)

    accuracy = 100 * correct / total
    return val_loss / len(val_loader), accuracy  


def test(model, test_loader, loss_fn, device):
    model.eval()
    test_loss, correct = 0, 0
    total = 0 
    model.eval()
    preds_list = []
    labels = []

    with torch.inference_mode():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            test_loss += loss_fn(y_pred, y).item()
            correct += (y_pred.argmax(dim=1) == y).sum().item()
            total += y.size(0)
            
            preds = y_pred.argmax(dim=1)
            preds_list.extend(preds.tolist())
            labels.extend(y.tolist())

    accuracy = 100 * correct / total
    return test_loss / len(test_loader), accuracy, torch.tensor(preds_list), torch.tensor(labels)


def print_classification_report(y, y_pred, class_names, filename):
    report = classification_report(y.cpu(), y_pred.cpu(), target_names=class_names, digits=3,  zero_division=0)
    
    print("Classification Report:")
    print(report)
    
    with open(filename, "w") as f:
        f.write(report)
    
    print(f"Report saved to {filename}")



def train_and_evaluate(model, train_loader, val_loader, test_loader, loss_fn, optimizer, device, epochs=10):

    total_training_time = 0
    best_val_acc = 0.0
    train_loss_values = []
    val_loss_values = []
    train_acc_values = []
    val_acc_values = []
    epoch_count = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        epoch_start_time = timer()
        
        train_loss, train_acc = train(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)

        print(f"Train Loss: {train_loss:.4f} | Train accuracy: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

            
        epoch_end_time = timer()
        epoch_duration = epoch_end_time - epoch_start_time
        total_training_time += epoch_duration
        print(f"Epoch {epoch+1} took {epoch_duration:.2f} seconds | Total training time so far: {total_training_time:.2} seconds")
        
        epoch_count.append(epoch)
        train_loss_values.append(train_loss)
        val_loss_values.append(val_loss)
        train_acc_values.append(train_acc)
        val_acc_values.append(val_acc)
    
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")

    plot_loss(epoch_count, train_loss_values, val_loss_values)
    plot_accuracy(epoch_count, train_acc_values, val_acc_values)
    
    test_loss, test_acc, preds_list, labels = test(model, test_loader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
    
    return preds_list, labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_classes = 26
model = CNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)


train_loader, val_loader, test_loader, class_names = load_data(
    train_dir="E:\\AN4\\licenta\\dataset_wave_complet\\train",
    val_dir="E:\\AN4\\licenta\\dataset_wave_complet\\val",
    test_dir="E:\\AN4\\licenta\\dataset_wave_complet\\test",
    batch_size=8
)


y_pred, labels = train_and_evaluate(model, train_loader, val_loader, test_loader, loss_fn, optimizer, device, epochs=10)

fig, ax = conf_mat(y_pred, labels, num_classes, class_names, figsize=(10,7))
plt.show()

print_classification_report(labels, y_pred, class_names, filename="classification_report_mel_cnn.txt")


