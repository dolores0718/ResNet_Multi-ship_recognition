import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D  # 用于三维散点图

# 设置音频文件夹路径
audio_folder = r'E:\毕业设计\船舶辐射噪声\inclusion_3000_exclusion_5000\train\audio'

# 定义类别标签
labels = ['background', 'cargo', 'passengership', 'tanker', 'tug']

# 定义每个类别的标记样式
markers = ['o', 's', '^', 'D', 'P']  # 分别为圆圈、方块、三角形、菱形、五角星

# 用来存储所有的STFT相位谱特征和对应的标签
all_features = []
all_labels = []

# 遍历每个文件夹
for label in labels:
    folder_path = os.path.join(audio_folder, label)
    # 获取文件夹中的音频文件列表，限制为前50个
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')][:50]

    for filename in files:
        audio_path = os.path.join(folder_path, filename)
        # 载入音频文件
        signal, sr = librosa.load(audio_path, sr=None)

        # 提取STFT特征
        stft = librosa.stft(signal)

        # 获取STFT的相位谱
        stft_phase = np.angle(stft)

        # 将STFT的相位谱展平为一维向量并添加到特征集合中
        stft_flat = stft_phase.flatten()
        all_features.append(stft_flat)

        # 将标签添加到标签集合中
        all_labels.append(label)

# 将特征和标签转换为numpy数组
X = np.array(all_features)
y = np.array(all_labels)

# --- PCA降维 ---
# 使用PCA将特征降到三维
pca = PCA(n_components=3)
X_pca_3d = pca.fit_transform(X)

# 使用PCA将特征降到二维
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X)

# --- 绘制三维特征散点图 ---
fig1 = plt.figure(figsize=(12, 10))
ax1 = fig1.add_subplot(111, projection='3d')

# 为每个类别绘制散点图
for i, label in enumerate(labels):
    indices = np.where(y == label)[0]  # 获取当前类别的索引
    X_class_3d = X_pca_3d[indices]  # 获取当前类别的PCA特征
    ax1.scatter(X_class_3d[:, 0], X_class_3d[:, 1], X_class_3d[:, 2], label=label, marker=markers[i], s=80)  # 使用不同的标记样式

# 设置三维图的标签和标题
ax1.set_xlabel('PCA1')
ax1.set_ylabel('PCA2')
ax1.set_zlabel('PCA3')
ax1.set_title('3D PCA of STFT Phase Features by Class')
ax1.legend(title="Classes", loc="upper right")

# 显示三维图
plt.show()

# --- 绘制二维特征散点图 ---
fig2 = plt.figure(figsize=(12, 10))
ax2 = fig2.add_subplot(111)

# 为每个类别绘制散点图
for i, label in enumerate(labels):
    indices = np.where(y == label)[0]  # 获取当前类别的索引
    X_class_2d = X_pca_2d[indices]  # 获取当前类别的PCA特征
    ax2.scatter(X_class_2d[:, 0], X_class_2d[:, 1], label=label, marker=markers[i], s=80)  # 使用不同的标记样式

# 设置二维图的标签和标题
ax2.set_xlabel('PCA1')
ax2.set_ylabel('PCA2')
ax2.set_title('2D PCA of STFT Phase Features by Class')
ax2.legend(title="Classes", loc="upper right")

# 显示二维图
plt.show()
