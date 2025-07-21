import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_path, label_path, transform=None):
        # 加载npz数据文件
        data = np.load(data_path)
        self.images = data['data']  # 假设数据文件包含键为'data'的数据
        self.labels = data['labels']  # 假设数据文件包含键为'labels'的数据

        # 标签编码
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# 2. 数据增强和预处理
transform = transforms.Compose([
    transforms.ToPILImage(),  # np.array转为PIL Image
    transforms.Resize((224, 224)),  # 调整大小到224x224
    transforms.ToTensor(),  # 转换为tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet的均值和标准差
])

# 3. 加载数据集
train_dataset = CustomDataset(
    data_path='/root/autodl-tmp/encoded_data/train_encoded.npz',
    label_path='/root/autodl-tmp/encoded_data/train_labels.npz', 
    transform=transform
)

test_dataset = CustomDataset(
    data_path='/root/autodl-tmp/encoded_data/test_encoded.npz',
    label_path='/root/autodl-tmp/encoded_data/test_labels.npz', 
    transform=transform
)

val_dataset = CustomDataset(
    data_path='/root/autodl-tmp/encoded_data/val_encoded.npz',
    label_path='/root/autodl-tmp/encoded_data/val_labels.npz', 
    transform=transform
)

# 4. 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 5. 加载ResNet18模型
model = models.resnet18(pretrained=True)

# 修改最后的全连接层（fc），适应5类输出
model.fc = nn.Linear(model.fc.in_features, 5)

# 6. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 7. 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 统计损失和准确率
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

# 8. 生成混淆矩阵
def generate_confusion_matrix(loader, model):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # 调试：输出前10个预测和真实标签
            print(f"Predicted: {predicted[:10]}, Labels: {labels[:10]}")  # 查看前10个预测和真实标签
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    return cm

# 9. 验证模型
val_cm = generate_confusion_matrix(val_loader, model)
print("Validation Confusion Matrix:\n", val_cm)  # 打印混淆矩阵

# 10. 绘制混淆矩阵
sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cargo', 'Tanker', 'Tug', 'PassengerShip', 'Others'], yticklabels=['Cargo', 'Tanker', 'Tug', 'PassengerShip', 'Others'])
plt.title('Validation Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 11. 测试模型
test_cm = generate_confusion_matrix(test_loader, model)
print("Test Confusion Matrix:\n", test_cm)  # 打印混淆矩阵

# 12. 绘制测试混淆矩阵
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cargo', 'Tanker', 'Tug', 'PassengerShip', 'Others'], yticklabels=['Cargo', 'Tanker', 'Tug', 'PassengerShip', 'Others'])
plt.title('Test Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 13. 保存模型
torch.save(model.state_dict(), 'resnet18_model.pth')
