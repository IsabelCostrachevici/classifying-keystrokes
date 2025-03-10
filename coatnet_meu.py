# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 23:13:58 2025

@author: user
"""
import librosa
import librosa.display
import torch
import torch.nn as nn
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


def conv_3x3_bn(inp, oup, image_size, downsample=False): 
    stride = 1 if not downsample else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )

class CoAtNet(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, num_classes=26):
        super().__init__()
        ih, iw = image_size
        self.s0 = self._make_layer(conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2))
        self.s1 = self._make_layer(conv_3x3_bn, channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4))
        self.s2 = self._make_layer(conv_3x3_bn, channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8))
        self.s3 = self._make_layer(conv_3x3_bn, channels[2], channels[3], num_blocks[3], (ih // 16, iw // 16))
        self.s4 = self._make_layer(conv_3x3_bn, channels[3], channels[4], num_blocks[4], (ih // 32, iw // 32))
        self.pool = nn.AvgPool2d(ih // 32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.pool(x).view(-1, x.shape[1])
        return self.fc(x)

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = [block(inp, oup, image_size, downsample=True)]
        layers += [block(oup, oup, image_size) for _ in range(depth - 1)]
        return nn.Sequential(*layers)

def coatnet_0():
    return CoAtNet((224, 224), 5, [2, 2, 3, 5, 2], [64, 96, 192, 384, 768], num_classes=26)


def load_spectrograms(base_folder, letter, segment_name):
    spectrogram_types = ["mel_spectrograms", "masked_mel_spectrograms", "mfcc_spectrograms",
                         "spectral_contrast_spectrograms", "wavelet_spectrograms"]
    
    spectrogram_tensors = []
    for spec_type in spectrogram_types:
        spec_path = os.path.join(base_folder, spec_type, letter, f"{segment_name}.png")
        if os.path.exists(spec_path):
            img = cv2.imread(spec_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            spectrogram_tensors.append(torch.tensor(img))
        else:
            print(f"Fișier lipsă: {spec_path}")  
            return None  

    return torch.stack(spectrogram_tensors, dim=0)

class KeystrokeDataset(Dataset):
    def __init__(self, root_folder, label_mapping):
        self.root_folder = root_folder
        self.classes = set()
        self.files = []

        spectrogram_types = ["mel_spectrograms", "masked_mel_spectrograms", "mfcc_spectrograms",
                             "spectral_contrast_spectrograms", "wavelet_spectrograms"]

        for spec_type in spectrogram_types:
            spec_folder = os.path.join(root_folder, spec_type)
            if os.path.exists(spec_folder):
                for letter in os.listdir(spec_folder):
                    letter_path = os.path.join(spec_folder, letter)
                    if os.path.isdir(letter_path):
                        self.classes.add(letter)  # Adăugăm litera în set
                        for file in os.listdir(letter_path):
                            if file.endswith(".png"):
                                segment_name = file.replace(".png", "")
                                self.files.append((letter, segment_name))

        self.classes = sorted(list(self.classes))
        print(f"Număr total de fișiere găsite: {len(self.files)}")

        if len(self.files) == 0:
            raise ValueError("Eroare: Nu s-au găsit fișiere de antrenare! Verifică structura folderului.")

        self.labels = [label_mapping[letter] for letter, _ in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        letter, segment_name = self.files[idx]
        img_tensor = load_spectrograms(self.root_folder, letter, segment_name)
        if img_tensor is None:
            return self.__getitem__((idx + 1) % len(self.files))  
        return img_tensor, torch.tensor(self.labels[idx], dtype=torch.long)



label_mapping = {chr(i): i - ord('a') for i in range(ord('a'), ord('z') + 1)}
dataset = KeystrokeDataset("E:\\AN4\\licenta\\dataset", label_mapping)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = coatnet_0().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")



def predict(model, base_folder, letter, segment_name):
    model.eval()
    img_tensor = load_spectrograms(base_folder, letter, segment_name)
    if img_tensor is None:
        print(f"Nu am găsit fișierul pentru {letter}/{segment_name}")
        return None
    img_tensor = img_tensor.unsqueeze(0).to(device)
    output = model(img_tensor)
    pred_label = torch.argmax(output, dim=1).item()
    return chr(pred_label + ord('a'))


print("Test Prediction:", predict(model, "E:\\AN4\\licenta\\dataset_testare", "b", "melspectrogram"))



