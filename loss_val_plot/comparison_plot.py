import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ================== 配置参数 ==================
models = [
    {'name': 'VGG16', 'file': 'VGG16 result.xlsx', 'color': ('#ff7f0e', '#ffbb78'), 'val_col': 'accuracy'},
    {'name': 'ResNet18', 'file': 'ResNet18 result.xlsx', 'color': ('#2ca02c', '#98df8a'), 'val_col': 'accuracy'},
    {'name': 'SE_ResNet', 'file': 'SE_ResNet.xlsx', 'color': ('#9467bd', '#c5b0d5'), 'val_col': 'accuracy'},
    {'name': '1D CNN', 'file': '1D CNN.xlsx', 'color': ('#1f77b4', '#aec7e8'), 'val_col': 'accuracy'}
]

# ================== 数据加载 ==================
def load_model_data(model_info):
    try:
        df = pd.read_excel(model_info['file'])
        df.columns = df.columns.str.strip().str.lower()  # Clean column names

        # Check if necessary columns are available
        required_cols = {'epoch', 'loss', model_info['val_col']}
        if not required_cols.issubset(df.columns):
            print(f"Warning: {model_info['name']} data lacks required columns: {required_cols - set(df.columns)}")
            return None

        return {
            'name': model_info['name'],
            'epochs': pd.to_numeric(df['epoch'], errors='coerce').fillna(0).astype(int),
            'loss': pd.to_numeric(df['loss'], errors='coerce').fillna(0),
            'val_acc': pd.to_numeric(df[model_info['val_col']], errors='coerce'),
            'color': model_info['color'][0],
            'val_color': model_info['color'][1]
        }
    except Exception as e:
        print(f"Error loading {model_info.get('name', 'unknown')}: {str(e)}")
        return None

all_data = [data for data in (load_model_data(m) for m in models) if data is not None]

# ================== Accuracy data from the first part ==================
# Use the accuracy data provided initially
dcnn_acc = np.array([53.68, 57.34, 62.83, 66.82, 69.29, 69.67, 71.22, 72.92, 70.63, 73.30,
                     72.49, 72.47, 72.21, 72.37, 72.47, 72.21, 73.14, 72.24, 71.18, 71.09])

vgg16_acc = np.array([62.45, 76.87, 82.72, 86.92, 89.57, 91.33, 92.99, 94.07, 95.06, 95.85,
                      96.61, 97.05, 97.39, 97.81, 98.14, 98.38, 98.47, 98.71, 98.86, 98.96])

resnet18_acc = np.array([71.01, 84.74, 89.44, 92.22, 94.16, 95.57, 96.63, 97.46, 98.04, 98.58,
                         98.93, 99.00, 99.13, 99.29, 99.32, 99.54, 99.53, 99.58, 99.55, 99.59])

se_resnet_acc = np.array([70.21, 84.04, 89.02, 91.78, 93.64, 94.83, 95.70, 96.40, 96.83, 97.24,
                          97.54, 97.88, 98.09, 98.18, 98.48, 98.54, 98.59, 98.73, 98.78, 98.90])

# Epochs
epochs = np.arange(1, 21)

# ================== 绘制双子图对比 ==================
plt.figure(figsize=(18, 6))  # Adjust canvas size

# ------------------ Loss子图 ------------------
plt.subplot(1, 2, 1)  # First subplot (Loss)
for data in all_data:
    plt.plot(data['epochs'], data['loss'],
             marker='o',
             color=data['color'],
             linewidth=2,
             markersize=6,
             label=f'{data["name"]} Loss')

plt.title('Training Loss Comparison', fontsize=14, pad=10)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right', fontsize=10)
plt.xticks(sorted(set().union(*[set(d['epochs']) for d in all_data])))  # Dynamically adjust x-axis ticks
plt.ylim(bottom=0)  # Set y-axis minimum to 0

# ------------------ Accuracy子图 ------------------
plt.subplot(1, 2, 2)  # Second subplot (Accuracy)
plt.plot(epochs, dcnn_acc, marker='o', linestyle='-', label='1D-CNN', color='#aec7e8')
plt.plot(epochs, vgg16_acc, marker='s', linestyle='-', label='VGG16', color='#ffbb78')
plt.plot(epochs, resnet18_acc, marker='^', linestyle='-', label='ResNet18', color='#98df8a')
plt.plot(epochs, se_resnet_acc, marker='d', linestyle='-', label='SE_ResNet', color='#c5b0d5')

plt.title('Validation Accuracy Comparison', fontsize=14, pad=10)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right', fontsize=10)
plt.xticks(epochs)
plt.ylim(50, 100)

# ------------------ 全局设置 ------------------
plt.tight_layout(pad=3.0)  # Add padding between subplots
plt.show()
