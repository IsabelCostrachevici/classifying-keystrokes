import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from noisereduce import reduce_noise
import soundfile as sf
import pywt
from PIL import Image

input_folder = "E:\\AN4\\licenta\\val_preprocesare\\Raw_Data"

base_output_folders = {
    "waveforms": "E:\\AN4\\licenta\\val_preprocesare\\waveforms",
    "segmented_audio": "E:\\AN4\\licenta\\val_preprocesare\\segmented_audio",
    "fft_representations": "E:\\AN4\\licenta\\val_preprocesare\\fft_representations",
    "mel_spectrograms": "E:\\AN4\\licenta\\val_preprocesare\\mel_spectrograms",    
    "mfcc_spectrograms": "E:\\AN4\\licenta\\val_preprocesare\\mfcc_spectrograms",
    "wavelet_spectrograms": "E:\\AN4\\licenta\\val_preprocesare\\wavelet_spectrograms"
}

for category, path in base_output_folders.items():
    os.makedirs(path, exist_ok=True)    

def wavelet_transform(segment, sr):
    scales = np.arange(1, 64)  
    wavelet = 'morl'  
    
    coefficients, _ = pywt.cwt(segment, scales, wavelet, 1.0 / sr)
    
    return coefficients

for file in os.listdir(input_folder):
    if file.endswith(".wav"):
        letter = os.path.splitext(file)[0]  
        audio_path = os.path.join(input_folder, file)
        y, sr = librosa.load(audio_path, sr=None)
        y = y / np.max(np.abs(y))  

        y_denoised = reduce_noise(y=y, sr=sr)

        onset_frames = librosa.onset.onset_detect(y=y_denoised, sr=sr, delta=0.2, pre_max=10, post_max=10, units="time")

        min_time_gap = 0.3
        filtered_onsets = [onset_frames[0]]
        for onset in onset_frames[1:]:
            if onset - filtered_onsets[-1] > min_time_gap:
                filtered_onsets.append(onset)

        fixed_length = 0.3  
        fixed_samples = int(fixed_length * sr)

        if len(filtered_onsets) > 2:
            filtered_onsets = filtered_onsets[:-1]

        for category, base_path in base_output_folders.items():
            letter_path = os.path.join(base_path, letter)
            os.makedirs(letter_path, exist_ok=True)

        for idx in range(len(filtered_onsets)-1):
            start = int(filtered_onsets[idx] * sr)
            end = int(filtered_onsets[idx + 1] * sr)
            segment = y[start:end]

            if len(segment) < fixed_samples:
                segment = np.pad(segment, (0, fixed_samples - len(segment)), mode='constant')
            else:
                segment = segment[:fixed_samples]

            segment_name = f"{letter}_keystroke_{idx + 1}"
            
            sf.write(os.path.join(base_output_folders["segmented_audio"], letter, f"{segment_name}.wav"), segment, sr)

            plt.figure(figsize=(10, 3))
            librosa.display.waveshow(segment, sr=sr, alpha=0.8)
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.tight_layout()
            plt.savefig(os.path.join(base_output_folders["waveforms"], letter, f"{segment_name}.png"))
            plt.close()

            fft_values = np.fft.fft(segment)
            fft_magnitude = np.abs(fft_values)
            fft_frequencies = np.fft.fftfreq(len(segment), d=1/sr)

            plt.figure(figsize=(10, 4))
            plt.plot(fft_frequencies[:len(fft_frequencies)//2], fft_magnitude[:len(fft_magnitude)//2])
            plt.xlabel("Frequency")
            plt.ylabel("Magnitude")
            plt.tight_layout()
            plt.savefig(os.path.join(base_output_folders["fft_representations"], letter, f"{segment_name}.png"))
            plt.close()
            
            mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

            plt.figure(figsize=(10, 3))
            librosa.display.specshow(mel_spectrogram_db, sr=sr, hop_length=512, x_axis = "time", y_axis='mel')
            plt.tight_layout()
            output_path = os.path.join(base_output_folders["mel_spectrograms"], letter, f"{segment_name}.png")
            plt.savefig(output_path, transparent=False, facecolor="white")
            plt.close()
            img = Image.open(output_path).convert("RGB")
            img.save(output_path)

            mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
            mfccs = mfccs[1:, :]

            plt.figure(figsize=(10, 3))
            librosa.display.specshow(mfccs, sr=sr, hop_length=512, x_axis ='time')
            plt.tight_layout()
            output_path=os.path.join(base_output_folders["mfcc_spectrograms"], letter, f"{segment_name}.png")
            plt.savefig(output_path, transparent=False, facecolor="white")
            plt.close()
            img = Image.open(output_path).convert("RGB")
            img.save(output_path)
            
            wavelet_coeffs = wavelet_transform(segment, sr)

            plt.figure(figsize=(10, 3))
            plt.imshow(np.abs(wavelet_coeffs), aspect='auto', extent=[0, fixed_length, 1, 64])
            plt.xlabel("Time")
            plt.ylabel("Scales")
            plt.tight_layout()
            output_path=os.path.join(base_output_folders["wavelet_spectrograms"], letter, f"{segment_name}.png")
            plt.savefig(output_path, transparent=False, facecolor="white")
            plt.close()
            img = Image.open(output_path).convert("RGB")
            img.save(output_path)