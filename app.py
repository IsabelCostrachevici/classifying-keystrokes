import streamlit as st
import uuid
from predict import predict, generate_image
import torch
import torch.nn as nn


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

st.set_page_config(page_title="Keystroke Recognition", layout="wide")

st.markdown("""
    <style>
        body {
            background-color: #f8f9fa;
        }
        .title {
            font-size: 42px;
            font-weight: bold;
            color: #800020;
            text-align: center;
            margin-top: 0px;
            margin-bottom: 20px;
        }
        .section-header {
            font-size: 26px;
            color: #A70D2A;
            margin-top: 30px;
        }
        .prediction-box {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            border: 2px solid #A70D2A;
            margin-top: 5px;
            font-size: 18px;
        }
        .upload-section, .settings-section {
            background-color: #A70D2A;
            padding: 1px;
            margin-bottom: 5px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Platformă de predicții pentru semnale acustice emise prin dactilografiere</div>', unsafe_allow_html=True)

with open("ro_words.txt", "r", encoding="utf-8") as f:
    word_list = set(line.strip().split()[0].lower() for line in f if line.strip())

class_names = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
    'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### Încarcă fișierul WAV")
uploaded_file = st.file_uploader("Încarcă fișierul audio:", type=["wav"])
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="settings-section">', unsafe_allow_html=True)
st.markdown("### Setările pentru model și reprezentare")
model_choice = st.selectbox("Alege modelul:", ["CoAtNet", "CNN"])
rep_type = st.selectbox("Alege tipul de reprezentare:", ["Melspectrogram", "MFCC", "Wavelet"])
st.markdown('</div>', unsafe_allow_html=True)

if st.button("Realizează predicția"):
    if uploaded_file is None:
        st.warning("Este necesară încărcarea unui fișier audio înaintea realizării predicției.")
    else:
        temp_audio_path = f"temp_audio_{uuid.uuid4().hex}.wav"
        with open(temp_audio_path, "wb") as f:
            f.write(uploaded_file.read())

        st.audio(temp_audio_path)

        if model_choice == "CoAtNet":
            model = coatnet_0().to(device)
            if rep_type == "Melspectrogram":
                model_path = "E:/AN4/licenta/best_model_mel_coatnet.pth"
            elif rep_type == "MFCC":
                model_path = "E:/AN4/licenta/best_model_mfcc_coatnet.pth"
            elif rep_type == "Wavelet":
                model_path = "E:/AN4/licenta/best_model_wave_coatnet.pth"
            else:
                st.error("Nu există acest tip de reprezentare pentru algoritmul CoAtNet")
                st.stop()
        elif model_choice == "CNN":
            model = CNN().to(device)
            if rep_type == "Melspectrogram":
                model_path = "E:/AN4/licenta/best_model_mel_cnn.pth"
            elif rep_type == "MFCC":
                model_path = "E:/AN4/licenta/best_model_mfcc_cnn.pth"
            elif rep_type == "Wavelet":
                model_path = "E:/AN4/licenta/best_model_wave_cnn.pth"
            else:
                st.error("Nu există acest tip de reprezentare pentru modelul CNN.")
                st.stop()
        else:
            st.error("Model nu există.")
            st.stop()

        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            st.error(f"Eroare de încărcare a modelului: {e}")
            st.stop()

        predicted_letters, corrected, all_top_predictions, top_7_letters_per_segment = predict(
            model=model,
            audio_path=temp_audio_path,
            rep_type=rep_type,
            class_names=class_names,
            device=device,
            word_list=word_list
        )
        
        st.markdown('<div class="section-header"> Predicția inițială</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="prediction-box">{"".join(predicted_letters)}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header"> Primele 7 predicții pentru fiecare literă</div>', unsafe_allow_html=True)
        for i, top7 in enumerate(top_7_letters_per_segment):
            st.markdown(f'<div class="prediction-box"><b>Litera {i+1}:</b> {", ".join(top7)}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header"> Top cuvinte alternative</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="prediction-box">{", ".join(all_top_predictions[:5])}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header"> Cuvântul final corectat</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="prediction-box">{corrected}</div>', unsafe_allow_html=True)