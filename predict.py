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
from itertools import product

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

def wavelet_transform(segment, sr):
    scales = np.arange(1, 64)
    wavelet = 'morl'
    
    coefficients, _ = pywt.cwt(segment, scales, wavelet, 1.0 / sr)
    
    return coefficients


def generate_image(segment, sr, rep_type, temp_img_path):
    if rep_type == "Melspectrogram":
        mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        plt.figure(figsize=(10, 3)) 
        librosa.display.specshow(mel_spec_db, sr=sr, hop_length=512)
        #plt.colorbar(format='%+2.0f dB')
        #plt.tight_layout()
        plt.savefig(temp_img_path, transparent=False, facecolor="white")
        plt.close()

    elif rep_type == "MFCC":
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
        mfccs = mfccs[1:, :]

        plt.figure(figsize=(10, 3))
        librosa.display.specshow(mfccs, sr=sr, hop_length=512)
        #plt.colorbar(format='%+2.0f dB')
        #plt.tight_layout()
        plt.savefig(temp_img_path, transparent=False, facecolor="white")
        plt.close()

    elif rep_type == "Wavelet":
        coeffs = wavelet_transform(segment, sr)
        
        plt.figure(figsize=(10, 3))
        plt.imshow(np.abs(coeffs), aspect='auto', extent=[0, 0.3, 1, 64])
        #plt.colorbar(label='Magnitude')
        #plt.xlabel("Time (s)")
        #plt.ylabel("Scales")
        #plt.tight_layout()
        plt.savefig(temp_img_path, transparent=False, facecolor="white")
        plt.close()

    img = Image.open(temp_img_path).convert("RGB")
    os.remove(temp_img_path)
    return img

def predict(model, audio_path, rep_type="Melspectrogram", class_names=None, device='cpu', word_list=None):
    model.eval()
    y, sr = librosa.load(audio_path, sr=None)
    y = y / np.max(np.abs(y))
    y_denoised = reduce_noise(y=y, sr=sr)

    onset_frames = librosa.onset.onset_detect(y=y_denoised, sr=sr, delta=0.2, pre_max=10, post_max=10, units="time")

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
    all_top_probs = []
    all_top_indices = []
    predicted_letters = []
    top_7_letters_per_segment = []

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
        img = generate_image(segment, sr, rep_type, temp_img_path)
        input_tensor = transform(img).unsqueeze(0).to(device)

        img.save(f"letter_{idx}.png")
        
        with torch.inference_mode():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).squeeze()
            top_probs, top_indices = torch.topk(probs, k=min(7, len(probs)))

            all_top_probs.append(probs.cpu().numpy())
            all_top_indices.append(top_indices.cpu().numpy())

            pred_idx = top_indices[0].item()
            predicted_letter = class_names[pred_idx] if class_names else str(pred_idx)
            predicted_letters.append(predicted_letter)

            top_letters = [class_names[i] for i in top_indices]
            top_7_letters_per_segment.append(top_letters)

            print(f"Segment {idx}: Predicted Letter = {predicted_letter}, Top-7 = {top_letters}")

    print("\nInitial Prediction:", ''.join(predicted_letters))

    candidates = list(product(*all_top_indices))
    words = [''.join(class_names[i] for i in word) for word in candidates]
    word_scores = [np.prod([probs[i] for probs, i in zip(all_top_probs, word)]) for word in candidates]

    if word_list:
        filtered = [(w, s) for w, s in zip(words, word_scores) if w in word_list]
        filtered.sort(key=lambda x: x[1], reverse=True)
        corrected = filtered[0][0] if filtered else words[np.argmax(word_scores)]
        print("Top candidate matches:")
        for cand, score in filtered[:5]:
            print(f"{cand}")
        all_top_predictions = [w for w, s in filtered]
    else:
        corrected = words[np.argmax(word_scores)]
        all_top_predictions = words

    print("Corrected Word:", corrected)
    return predicted_letters, corrected, all_top_predictions, top_7_letters_per_segment



with open("ro_words.txt", "r", encoding="utf-8") as f:
    custom_word_list = set(line.strip().split()[0].lower() for line in f if line.strip())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
model.load_state_dict(torch.load("E:\\AN4\\licenta\\best_model_mel_cnn.pth", map_location=device))

class_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']  

audio_file = "E:\\AN4\\licenta\\parola_3.wav"
predicted_word = predict(model, audio_file, rep_type="Melspectrogram", class_names=class_names, device=device, word_list=custom_word_list)
print("Predicted Word:", predicted_word)