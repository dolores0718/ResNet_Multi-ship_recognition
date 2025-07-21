import matplotlib.pyplot as plt
import numpy as np

# 数据
dropout_rates = np.array([0.3, 0.4, 0.5, 0.6])
accuracy_SGD = np.array([64.30, 69.42, 67.65, 52.36])
accuracy_SGD_Momentum = np.array([82.54, 81.56, 80.20, 79.96])
accuracy_RMSProp = np.array([94.50, 93.26, 92.48, 91.86])
accuracy_Adam = np.array([96.50, 95.68, 95.40, 94.76])

# 颜色选择
colors = [
    '#4477AA',  # 深海蓝（SGD）
    '#EE7733',  # 暖橙色（SGD+Momentum）
    '#009988',  # 蓝绿色（RMSProp）
    '#BB5566'   # 玫瑰红（Adam）
]
markers = ['o', 'D', 's', '^']  # 圆圈, 菱形, 正方形, 三角形
labels = ['SGD', 'SGD with Momentum', 'RMSProp', 'Adam']
accuracies = [accuracy_SGD, accuracy_SGD_Momentum, accuracy_RMSProp, accuracy_Adam]

# 绘制散点图
plt.figure(figsize=(8, 6))
for i in range(4):
    plt.scatter(dropout_rates, accuracies[i], marker=markers[i], color=colors[i], label=labels[i])

# 添加标签和标题
plt.xlabel('Dropout Rate')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Dropout Rate for Different Optimizers')
plt.legend()

# 显示图像
plt.show()