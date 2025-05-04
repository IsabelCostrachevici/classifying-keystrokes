# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 10:19:56 2025

@author: user
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, random_split, DataLoader
from torchvision import datasets, transforms
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
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.1)
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
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.1)            
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
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2)
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
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2)
            
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.3)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.4)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
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
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def split(train_dir, val_dir, test_dir, batch_size=8, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    full_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
    total_size = len(full_dataset)

    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    new_train, new_val, new_test = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(new_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(new_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(new_test, batch_size=batch_size, shuffle=False)

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


def print_report(y, y_pred, class_names, filename):
    report = classification_report(y.cpu(), y_pred.cpu(), target_names=class_names, digits=3,  zero_division=0)
    
    print("Classification Report:")
    print(report)
    
    with open(filename, "w") as f:
        f.write(report)
    
    print(f"Report saved to {filename}")


def train_and_evaluate(model, train_loader, val_loader, test_loader, loss_fn, optimizer, device, epochs=30):

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
            torch.save(model.state_dict(), "best_model_noise_split_cnn.pth")
   
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


train_loader, val_loader, test_loader, class_names = split(
    train_dir="E:\\AN4\\licenta\\dataset_mel_noise\\train",
    val_dir="E:\\AN4\\licenta\\dataset_mel_noise\\val",
    test_dir="E:\\AN4\\licenta\\dataset_mel_noise\\test",
    batch_size=8
)


y_pred, labels = train_and_evaluate(model, train_loader, val_loader, test_loader, loss_fn, optimizer, device, epochs=50)

fig, ax = conf_mat(y_pred, labels, num_classes, class_names, figsize=(10,7))
plt.show()

print_report(labels, y_pred, class_names, filename="classification_report_noise__split_cnn.txt")