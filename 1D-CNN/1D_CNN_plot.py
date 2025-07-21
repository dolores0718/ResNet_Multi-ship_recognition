import matplotlib.pyplot as plt

# 数据准备
epochs = list(range(1, 21))
loss = [1.2804, 1.0683, 0.9518, 0.8456, 0.7577, 0.6754, 0.5941, 0.5345, 0.4805, 0.4326,
        0.3842, 0.3351, 0.3094, 0.2592, 0.2276, 0.2200, 0.1674, 0.1465, 0.1374, 0.1133]
accuracy = [53.68, 57.34, 62.83, 66.82, 69.29, 69.67, 71.22, 72.92, 70.63, 73.30,
            72.49, 72.47, 72.21, 72.37, 72.47, 72.21, 73.14, 72.24, 71.18, 71.09]

# 创建画布 (与示例完全相同的尺寸)
plt.figure(figsize=(12, 6))

# 绘制损失曲线 (左图)
plt.subplot(1, 2, 1)  # 1行2列的第1个位置
plt.plot(epochs, loss,
         marker='o',         # 数据点标记
         color='#1f77b4',    # 标准蓝色
         label='Loss')       # 图例标签

# 统一格式设置
plt.title('Training Loss over Epochs')  # 标题与示例一致
plt.xlabel('Epoch')                    # X轴标签
plt.ylabel('Loss')                     # Y轴标签
plt.grid(True, linestyle='--', alpha=0.7)  # 半透明虚线网格
plt.legend(loc='upper right')          # 图例位置

# 绘制准确率曲线 (右图)
plt.subplot(1, 2, 2)  # 1行2列的第2个位置
plt.plot(epochs, accuracy,
         marker='o',         # 保持相同标记样式
         color='#aec7e8',    # 浅蓝色
         label='Accuracy')   # 图例标签

# 统一格式设置
plt.title('Validation Accuracy over Epochs')  # 标题格式匹配
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')    # 添加百分比单位
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right')  # 适合准确率的图例位置

# 优化布局
plt.tight_layout(pad=3.0)  # 增加子图间距，防止标签重叠

# 显示图像
plt.show()