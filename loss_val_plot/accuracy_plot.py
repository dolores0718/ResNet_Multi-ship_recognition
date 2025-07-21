import matplotlib.pyplot as plt
import numpy as np

# 1D-CNN accuracy
dcnn_acc = np.array([
    53.68, 57.34, 62.83, 66.82, 69.29, 69.67, 71.22, 72.92, 70.63, 73.30,
    72.49, 72.47, 72.21, 72.37, 72.47, 72.21, 73.14, 72.24, 71.18, 71.09
])

# VGG16 accuracy
vgg16_acc = np.array([
    62.45, 76.87, 82.72, 86.92, 89.57, 91.33, 92.99, 94.07, 95.06, 95.85,
    96.61, 97.05, 97.39, 97.81, 98.14, 98.38, 98.47, 98.71, 98.86, 98.96
])

# ResNet18 accuracy
resnet18_acc = np.array([
    71.01, 84.74, 89.44, 92.22, 94.16, 95.57, 96.63, 97.46, 98.04, 98.58,
    98.93, 99.00, 99.13, 99.29, 99.32, 99.54, 99.53, 99.58, 99.55, 99.59
])

# SE_ResNet accuracy
se_resnet_acc = np.array([
    70.21, 84.04, 89.02, 91.78, 93.64, 94.83, 95.70, 96.40, 96.83, 97.24,
    97.54, 97.88, 98.09, 98.18, 98.48, 98.54, 98.59, 98.73, 98.78, 98.90
])

# Epochs
epochs = np.arange(1, 21)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, dcnn_acc, marker='o', linestyle='-', label='1D-CNN', color='#1f77b4')
plt.plot(epochs, vgg16_acc, marker='s', linestyle='-', label='VGG16', color='#ff7f0e')
plt.plot(epochs, resnet18_acc, marker='^', linestyle='-', label='ResNet18', color='#2ca02c')
plt.plot(epochs, se_resnet_acc, marker='d', linestyle='-', label='SE_ResNet', color='#9467bd')

plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Over Epochs')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(epochs)
plt.ylim(50, 100)

plt.show()