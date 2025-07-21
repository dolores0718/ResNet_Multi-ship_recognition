import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# 设置音频文件路径
audio_path = r'E:\毕业设计\船舶辐射噪声\inclusion_3000_exclusion_5000\train\audio\cargo\198.wav'

# 加载音频文件
signal, sr = librosa.load(audio_path, sr=None)

# --- 绘制时域波形图 ---
plt.figure(figsize=(10, 6))
librosa.display.waveshow(signal, sr=sr)
plt.title('Time-domain Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# --- 计算log-Mel谱 ---
log_mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
log_mel = librosa.power_to_db(log_mel, ref=np.max)

# 绘制log-Mel谱图
plt.figure(figsize=(10, 6))
librosa.display.specshow(log_mel, x_axis='time', y_axis='mel', sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title('Log-Mel Spectrogram')
plt.show()

# --- 计算MFCC ---
mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)

# 绘制MFCC图
plt.figure(figsize=(10, 6))
librosa.display.specshow(mfcc, x_axis='time', sr=sr)
plt.colorbar()
plt.title('MFCC')
plt.show()

# --- 计算STFT ---
stft = librosa.stft(signal)
stft_magnitude = np.abs(stft)
stft_phase = np.angle(stft)

# --- 绘制STFT幅度谱图 ---
plt.figure(figsize=(10, 6))
librosa.display.specshow(librosa.amplitude_to_db(stft_magnitude, ref=np.max), y_axis='log', x_axis='time', sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title('STFT Magnitude Spectrum')
plt.show()

# --- 绘制STFT相位谱图 ---
plt.figure(figsize=(10, 6))
librosa.display.specshow(stft_phase, y_axis='log', x_axis='time', sr=sr)
plt.colorbar()
plt.title('STFT Phase Spectrum')
plt.show()
