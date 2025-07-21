import os
import numpy as np
import librosa
import cv2

# 设置音频文件路径
base_dir = '/root/autodl-tmp/inclusion_3000_exclusion_5000'

# 处理音频文件的函数
def process_audio_file(audio_path):
    # 读取音频文件
    y, sr = librosa.load(audio_path, sr=None)

    # 提取 STFT、Log-Mel 和 MFCC 特征
    D = np.abs(librosa.stft(y))  # STFT 特征
    log_mel = librosa.feature.melspectrogram(y=y, sr=sr)  # Log-Mel 特征
    mfcc = librosa.feature.mfcc(y=y, sr=sr)  # MFCC 特征

    # 选择目标形状 (224, 224) 
    target_shape = (224, 224)  # 目标形状：224x224

    # 对每个特征进行插值，使其时间维度和频率维度都缩放为目标形状
    D_resized = cv2.resize(D, target_shape, interpolation=cv2.INTER_LINEAR)
    log_mel_resized = cv2.resize(log_mel, target_shape, interpolation=cv2.INTER_LINEAR)
    mfcc_resized = cv2.resize(mfcc, target_shape, interpolation=cv2.INTER_LINEAR)

    # 将它们堆叠成多通道输入 (T, F, 3)，即 (224, 224, 3)
    input_data = np.stack((D_resized, log_mel_resized, mfcc_resized), axis=-1)

    return input_data

# 遍历数据集并处理
def process_dataset(base_dir, subset='train'):
    categories = ['background', 'cargo', 'tanker', 'tug', 'passengership']
    data = []
    labels = []

    subset_dir = os.path.join(base_dir, subset, 'audio')

    for category in categories:
        category_path = os.path.join(subset_dir, category)
        for audio_file in os.listdir(category_path):
            if audio_file.endswith('.wav'):
                audio_path = os.path.join(category_path, audio_file)
                
                try:
                    processed_data = process_audio_file(audio_path)
                    
                    # 检查数据形状，确保每个文件的形状一致
                    if processed_data.shape != (224, 224, 3):
                        print(f"Skipping {audio_file} due to unexpected shape: {processed_data.shape}")
                        continue
                    
                    data.append(processed_data)
                    labels.append(category)
                    
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
                    continue

    return np.array(data), np.array(labels)


# 调用函数处理数据集
train_data, train_labels = process_dataset(base_dir, subset='train')
test_data, test_labels = process_dataset(base_dir, subset='test')
validation_data, validation_labels = process_dataset(base_dir, subset='validation')

# 输出数据形状确认是否正确
print("训练集数据形状:", train_data.shape)  # (样本数, 224, 224, 3)
print("训练集标签形状:", train_labels.shape)  # (样本数,)
print("测试集数据形状:", test_data.shape)
print("测试集标签形状:", test_labels.shape)
print("验证集数据形状:", validation_data.shape)
print("验证集标签形状:", validation_labels.shape)

# 保存数据和标签为 .npz 格式
np.savez('/root/autodl-tmp/train_data_labels.npz', data=train_data, labels=train_labels)
np.savez('/root/autodl-tmp/test_data_labels.npz', data=test_data, labels=test_labels)
np.savez('/root/autodl-tmp/validation_data_labels.npz', data=validation_data, labels=validation_labels)

# 你可以查看文件是否成功保存
print("数据已保存为 /root/autodl-tmp/train_data_labels.npz, /root/autodl-tmp/test_data_labels.npz, /root/autodl-tmp/validation_data_labels.npz")
