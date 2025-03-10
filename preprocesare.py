# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 21:50:24 2024

@author: user
"""
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from noisereduce import reduce_noise
import soundfile as sf
import pywt

# Folder cu înregistrări audio
input_folder = "E:\\AN4\\licenta\\dataset_bun\\Raw_Data"

# Foldere de ieșire pentru fiecare categorie de date
base_output_folders = {
    "waveforms": "E:\\AN4\\licenta\\dataset_bun\\waveforms",
    "segmented_audio": "E:\\AN4\\licenta\\dataset_bun\\segmented_audio",
    "fft_representations": "E:\\AN4\\licenta\\dataset_bun\\fft_representations",
    "mel_spectrograms": "E:\\AN4\\licenta\\dataset_bun\\mel_spectrograms",
    "csv_mel_spectrograms": "E:\\AN4\\licenta\\dataset_bun\\csv_mel_spectrograms",
    "masked_mel_spectrograms": "E:\\AN4\\licenta\\dataset_bun\\masked_mel_spectrograms",
    "csv_masked_mel_spectrograms": "E:\\AN4\\licenta\\dataset_bun\\csv_masked_mel_spectrograms",
    "mfcc_spectrograms": "E:\\AN4\\licenta\\dataset_bun\\mfcc_spectrograms",
    "csv_mfcc_spectrograms": "E:\\AN4\\licenta\\dataset_bun\\csv_mfcc_spectrograms",
    "spectral_contrast_spectrograms": "E:\\AN4\\licenta\\dataset_bun\\spectral_contrast_spectrograms",
    "csv_spectral_contrast_spectrograms": "E:\\AN4\\licenta\\dataset_bun\\csv_spectral_contrast_spectrograms",
    "wavelet_spectrograms": "E:\\AN4\\licenta\\dataset_bun\\wavelet_spectrograms",
    "csv_wavelet_spectrograms": "E:\\AN4\\licenta\\dataset_bun\\csv_wavelet_spectrograms"
}

# Crearea folderelor dacă nu există
for category, path in base_output_folders.items():
    os.makedirs(path, exist_ok=True)
    
    
def frequency_masking(spectrogram, num_masks=1, mask_size=20):
    augmented_spectrogram = spectrogram.copy()
    num_mels, time_steps = augmented_spectrogram.shape
    for _ in range(num_masks):
        f0 = np.random.randint(0, num_mels - mask_size)
        augmented_spectrogram[f0:f0 + mask_size, :] = 0  
    return augmented_spectrogram

# Transformata wavelet
def compute_wavelet_transform(segment, sr):
    scales = np.arange(1, 64)  # Mai puține scale pentru viteză crescută
    wavelet = 'morl'  # Morlet Wavelet
    
    coefficients, _ = pywt.cwt(segment, scales, wavelet, 1.0 / sr)
    
    return coefficients

for file in os.listdir(input_folder):
    if file.endswith(".wav"):
        letter = os.path.splitext(file)[0]  
        audio_path = os.path.join(input_folder, file)
        y, sr = librosa.load(audio_path, sr=None)
        y = y / np.max(np.abs(y))  

        # Reducerea zgomotului
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

        # Creare subfoldere pentru litera curentă în fiecare categorie
        for category, base_path in base_output_folders.items():
            letter_path = os.path.join(base_path, letter)
            os.makedirs(letter_path, exist_ok=True)

        # Segmentarea audio
        for idx in range(len(filtered_onsets)-1):
            start = int(filtered_onsets[idx] * sr)
            end = int(filtered_onsets[idx + 1] * sr)
            segment = y[start:end]

            if len(segment) < fixed_samples:
                segment = np.pad(segment, (0, fixed_samples - len(segment)), mode='constant')
            else:
                segment = segment[:fixed_samples]

            segment_name = f"{letter}_keystroke_{idx + 1}"

            # Salvarea segmentelor audio
            sf.write(os.path.join(base_output_folders["segmented_audio"], letter, f"{segment_name}.wav"), segment, sr)

            # Plot waveform
            plt.figure(figsize=(10, 3))
            librosa.display.waveshow(segment, sr=sr, alpha=0.8)
            plt.title(f"{segment_name} - Waveform")
            plt.tight_layout()
            plt.savefig(os.path.join(base_output_folders["waveforms"], letter, f"{segment_name}.png"))
            plt.close()

            # FFT
            fft_values = np.fft.fft(segment)
            fft_magnitude = np.abs(fft_values)
            fft_frequencies = np.fft.fftfreq(len(segment), d=1/sr)

            plt.figure(figsize=(10, 4))
            plt.plot(fft_frequencies[:len(fft_frequencies)//2], fft_magnitude[:len(fft_magnitude)//2])
            plt.title(f"{segment_name} - FFT")
            plt.tight_layout()
            plt.savefig(os.path.join(base_output_folders["fft_representations"], letter, f"{segment_name}.png"))
            plt.close()
            
            # Melspectrograma
            mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

            plt.figure(figsize=(10, 3))
            librosa.display.specshow(mel_spectrogram_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"{segment_name} - Mel Spectrogram")
            plt.tight_layout()
            plt.savefig(os.path.join(base_output_folders["mel_spectrograms"], letter, f"{segment_name}.png"))
            plt.close()
            np.savetxt(os.path.join(base_output_folders["csv_mel_spectrograms"], letter, f"{segment_name}.csv"), mel_spectrogram_db, delimiter=",")

            # Melspectrograma mascata
            masked_mel_spectrogram_db = frequency_masking(mel_spectrogram_db, num_masks=2, mask_size=15)
            plt.figure(figsize=(10, 3))
            librosa.display.specshow(masked_mel_spectrogram_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"{segment_name} - Masked Mel Spectrogram")
            plt.tight_layout()
            plt.savefig(os.path.join(base_output_folders["masked_mel_spectrograms"], letter, f"{segment_name}.png"))
            plt.close()
            np.savetxt(os.path.join(base_output_folders["csv_masked_mel_spectrograms"], letter, f"{segment_name}.csv"), masked_mel_spectrogram_db, delimiter=",")

            # MFCC
            mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
            mfccs_db = librosa.power_to_db(mfccs, ref=np.max)

            plt.figure(figsize=(10, 3))
            librosa.display.specshow(mfccs_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"{segment_name} - MFCC Spectrogram")
            plt.tight_layout()
            plt.savefig(os.path.join(base_output_folders["mfcc_spectrograms"], letter, f"{segment_name}.png"))
            plt.close()
            np.savetxt(os.path.join(base_output_folders["csv_mfcc_spectrograms"], letter, f"{segment_name}.csv"), mfccs_db, delimiter=",")

            # Contrast spectral
            spectral_contrast = librosa.feature.spectral_contrast(y=segment, sr=sr, n_fft=2048, hop_length=512, n_bands=6)

            plt.figure(figsize=(10, 3))
            librosa.display.specshow(spectral_contrast, x_axis='time', sr=sr, hop_length=512)
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"{segment_name} - Spectral Contrast")
            plt.tight_layout()
            plt.savefig(os.path.join(base_output_folders["spectral_contrast_spectrograms"], letter, f"{segment_name}.png"))
            plt.close()
            np.savetxt(os.path.join(base_output_folders["csv_spectral_contrast_spectrograms"], letter, f"{segment_name}.csv"), spectral_contrast, delimiter=",")
            
            # Wavelet
            wavelet_coeffs = compute_wavelet_transform(segment, sr)

            plt.figure(figsize=(10, 3))
            plt.imshow(np.abs(wavelet_coeffs), aspect='auto', cmap='viridis', extent=[0, fixed_length, 1, 64])
            plt.colorbar(label='Magnitude')
            plt.title(f"{segment_name} - Wavelet Spectrogram")
            plt.xlabel("Time (s)")
            plt.ylabel("Scales")
            plt.tight_layout()
            plt.savefig(os.path.join(base_output_folders["wavelet_spectrograms"], letter, f"{segment_name}.png"))
            plt.close()

            np.savetxt(os.path.join(base_output_folders["csv_wavelet_spectrograms"], letter, f"{segment_name}.csv"), np.abs(wavelet_coeffs), delimiter=",")


print("Processing complete. All files saved in designated folders.")


#plt.figure(figsize=(12, 4))
#librosa.display.waveshow(y_denoised, sr=sr, alpha=0.8)
#plt.vlines(filtered_onsets, -1, 1, color='r', linestyle='--', label='Onsets')
#plt.legend()
#plt.show()

