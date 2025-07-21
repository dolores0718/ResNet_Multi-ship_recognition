import os
import librosa
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# 读取音频文件并提取特征
def extract_features(audio_path, feature_type='stft'):
    y, sr = librosa.load(audio_path, sr=None)

    if feature_type == 'mfcc':
        features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(features, axis=1)

    elif feature_type == 'log-mel':
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        log_mel = librosa.power_to_db(mel_spectrogram)
        return np.mean(log_mel, axis=1)

    elif feature_type == 'stft':
        stft = librosa.stft(y)
        magnitude, _ = librosa.magphase(stft)
        return np.mean(magnitude, axis=1)


# 加载数据
def load_data(base_path, feature_type='stft'):
    features = []
    labels = []
    target_labels = ['background', 'cargo', 'tanker', 'tug', 'passengership']

    # 遍历 'audio' 文件夹下的类别文件夹
    audio_folder_path = os.path.join(base_path, 'audio')
    print(f"Loading files from: {audio_folder_path}")  # Debugging line: check the folder path

    if not os.path.exists(audio_folder_path):
        print(f"Folder does not exist: {audio_folder_path}")
        return np.array(features), np.array(labels)  # If folder does not exist, return empty lists

    for label in target_labels:
        # 在 'audio' 文件夹内加载每个类别
        folder_path = os.path.join(audio_folder_path, label)
        print(f"Loading files from: {folder_path}")  # Debugging line: check the folder path

        if not os.path.exists(folder_path):
            print(f"Folder does not exist: {folder_path}")
            continue  # Skip if folder doesn't exist

        for file_name in os.listdir(folder_path):
            if file_name.endswith('.wav'):
                audio_path = os.path.join(folder_path, file_name)
                feature = extract_features(audio_path, feature_type)
                features.append(feature)
                labels.append(label)

    return np.array(features), np.array(labels)


# 绘制混淆矩阵
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Reference')
    plt.title('Confusion Matrix')
    plt.show()


# 主程序
def main():
    # 数据集路径
    data_path = r'E:\毕业设计\船舶辐射噪声\inclusion_3000_exclusion_5000'

    # 确认文件夹路径
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    validation_path = os.path.join(data_path, 'validation')

    # 加载训练集和测试集
    X_train, y_train = load_data(train_path)
    X_test, y_test = load_data(test_path)

    # 训练SVM分类器
    svm_model = SVC(kernel='rbf',gamma='scale',C=1)  # 使用RBF核，尝试其他核函数可调整
    svm_model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = svm_model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred, labels=['background', 'cargo', 'tanker', 'tug', 'passengership'])
    plot_confusion_matrix(cm, labels=['background', 'cargo', 'tanker', 'tug', 'passengership'])

    # 分类报告
    print('Classification Report:')
    print(classification_report(y_test, y_pred, target_names=['background', 'cargo', 'tanker', 'tug', 'passengership']))


# 执行主程序
if __name__ == "__main__":
    main()
