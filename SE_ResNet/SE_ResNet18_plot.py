import pandas as pd
import matplotlib.pyplot as plt


file_path = 'SE_ResNet.xlsx'

# 读取Excel文件
df = pd.read_excel(file_path)


# 提取训练轮次（Epoch）、损失（Loss）和验证准确率（Validation Accuracy）
epochs = df['epoch']
loss = df['loss']
val_acc = df['accuracy']

# 绘制损失曲线
plt.figure(figsize=(12, 6))

# 绘制训练损失曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, loss, marker='o', label='Loss', color='#9467bd')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# 绘制验证准确率曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, val_acc, marker='o', color='#c5b0d5', label='Validation Accuracy')
plt.title('Validation Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.grid(True)
plt.legend()

# 调整布局
plt.tight_layout()
plt.show()
