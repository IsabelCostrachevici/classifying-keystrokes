import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer 
from plot import plot_loss
from plot import plot_accuracy
from conf_matrix import conf_mat
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from torch.utils.data import ConcatDataset, random_split, DataLoader
from torchvision import datasets, transforms


def conv_3x3_bn(inp, oup, image_size, downsample=False):
    stride = 1 if not downsample else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )

class MBConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if not downsample else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
            )
        
    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)

class CoAtNet(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, num_classes=26):
        super().__init__()
        ih, iw = image_size
        self.s0 = self._make_layer(conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2))
        self.s1 = self._make_layer(MBConv, channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4))
        self.s2 = self._make_layer(MBConv, channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8))
        self.s3 = self._make_layer(MBConv, channels[2], channels[3], num_blocks[3], (ih // 16, iw // 16))
        self.s4 = self._make_layer(MBConv, channels[3], channels[4], num_blocks[4], (ih // 32, iw // 32))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)

def coatnet_0():
    #num_blocks = [2, 2, 3, 5, 2]      
    num_blocks = [2, 2, 2, 2, 1]      
    channels = [64, 96, 192, 384, 768]        
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=26)


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

def print_classification_report(y, y_pred, class_names, filename):
    report = classification_report(y.cpu(), y_pred.cpu(), target_names=class_names, digits=3)
    
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
            torch.save(model.state_dict(), "best_model_mel_coatnet_noise_split_3.pth")

            
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
model = coatnet_0().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay = 0.0001)


train_loader, val_loader, test_loader, class_names = split(
    train_dir="E:\\AN4\\licenta\\dataset_mel_noise\\train",
    val_dir="E:\\AN4\\licenta\\dataset_mel_noise\\val",
    test_dir="E:\\AN4\\licenta\\dataset_mel_noise\\test",
    batch_size=8
)

y_pred, labels = train_and_evaluate(model, train_loader, val_loader, test_loader, loss_fn, optimizer, device, epochs=10)

fig, ax = conf_mat(y_pred, labels, num_classes, class_names, figsize=(10,7))
plt.show()

print_classification_report(labels, y_pred, class_names, filename="classification_report_mel_coatnet_noise_split_3.txt")