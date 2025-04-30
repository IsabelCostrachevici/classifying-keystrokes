# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 21:31:07 2025

@author: user
"""

import librosa
import librosa.display
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
from noisereduce import reduce_noise
import uuid
import torch
import torch.nn as nn

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
        

def wavelet_transform(segment, sr):
    scales = np.arange(1, 64)
    wavelet = 'morl'
    coefficients, _ = pywt.cwt(segment, scales, wavelet, 1.0 / sr)
    return coefficients

def generate_spectrogram(segment, sr, rep_type, temp_img_path):
    if rep_type == "mel":
        mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        plt.figure(figsize=(10, 3)) 
        librosa.display.specshow(mel_spec_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.savefig(temp_img_path, transparent=False, facecolor="white")
        plt.close()
        
    elif rep_type == "mfcc":
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
        mfccs = mfccs[1:, :]

        plt.figure(figsize=(10, 3))
        librosa.display.specshow(mfccs, sr=sr, hop_length=512, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.savefig(temp_img_path, transparent=False, facecolor="white")
        plt.close()

    elif rep_type == "wavelet":
        coeffs = wavelet_transform(segment, sr)
        plt.figure(figsize=(10, 3))
        plt.imshow(np.abs(coeffs), aspect='auto', extent=[0, 0.3, 1, 64])
        plt.colorbar(label='Magnitude')
        plt.xlabel("Time (s)")
        plt.ylabel("Scales")
        plt.tight_layout()
        plt.savefig(temp_img_path, transparent=False, facecolor="white")
        plt.close()

    img = Image.open(temp_img_path).convert("RGB")
    os.remove(temp_img_path)
    return img

def predict_word_from_audio(model, audio_path, rep_type="mel", class_names=None, device='cpu'):
    model.eval()
    y, sr = librosa.load(audio_path, sr=None)
    y = y / np.max(np.abs(y))
    y_denoised = reduce_noise(y=y, sr=sr)

    onset_frames = librosa.onset.onset_detect(y=y_denoised, sr=sr, delta=0.2, pre_max=10, post_max=10, units="time")

    # Add last onset if needed to reach end of signal
    audio_duration = librosa.get_duration(y=y, sr=sr)
    if len(onset_frames) == 0 or onset_frames[-1] < audio_duration - 0.1:
        onset_frames = np.append(onset_frames, audio_duration)

    min_time_gap = 0.3
    filtered_onsets = [onset_frames[0]]
    for onset in onset_frames[1:]:
        if onset - filtered_onsets[-1] > min_time_gap:
            filtered_onsets.append(onset)

    fixed_length = 0.3
    fixed_samples = int(fixed_length * sr)
    predictions = []

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])


    for idx in range(len(filtered_onsets) - 1):
        start = int(filtered_onsets[idx] * sr)
        end = int(filtered_onsets[idx + 1] * sr)
        segment = y[start:end]

        if len(segment) < fixed_samples:
            segment = np.pad(segment, (0, fixed_samples - len(segment)), mode='constant')
        else:
            segment = segment[:fixed_samples]

        temp_img_path = f"temp_{uuid.uuid4().hex}.png"
        img = generate_spectrogram(segment, sr, rep_type, temp_img_path)
        input_tensor = transform(img).unsqueeze(0).to(device)

        img.save(f"segment_{idx}.png")

        with torch.inference_mode():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            print(f"Segment {idx} probabilities:\n", probs.cpu().numpy())
            pred_idx = output.argmax(dim=1).item()
            predicted_letter = class_names[pred_idx] if class_names else str(pred_idx)
            predictions.append(predicted_letter)

    return "".join(predictions)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
model.load_state_dict(torch.load("E:\\AN4\\licenta\\best_model.pth", map_location=device))

class_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']  

audio_file = "E:\\AN4\\licenta\\pred.wav"
predicted_word = predict_word_from_audio(model, audio_file, rep_type="mel", class_names=class_names, device=device)
print("Predicted Word:", predicted_word)