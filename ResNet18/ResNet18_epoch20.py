import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

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

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

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

        # 存储预测和标签，用于后续计算指标
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    # 计算每个epoch的precision, recall和F1-score
    train_precision = precision_score(all_labels, all_preds, average=None)
    train_f1 = f1_score(all_labels, all_preds, average=None)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    print(f"Train Precision: {train_precision}")
    print(f"Train F1-Score: {train_f1}")

    # 计算并打印classification report
    print("Classification Report for Training Epoch:")
    print(classification_report(all_labels, all_preds, target_names=["background", "cargo", "tanker", "tug", "passengership"]))

    # 验证模型
    model.eval()
    correct = 0
    total = 0
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 存储预测和标签
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = 100 * correct / total
    print(f'Validation Accuracy: {val_acc:.2f}%')

    # 计算并打印classification report for validation
    print("Classification Report for Validation:")
    print(classification_report(val_labels, val_preds, target_names=["background", "cargo", "tanker", "tug", "passengership"]))

# 8. 测试模型
model.eval()
correct = 0
total = 0
test_preds = []
test_labels = []
test_loss = 0.0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 存储预测和标签
        test_preds.extend(predicted.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_acc = 100 * correct / total
test_loss /= len(test_loader)
print(f'Test Accuracy: {test_acc:.2f}%')
print(f'Test Loss: {test_loss:.4f}')

# 计算并打印classification report for test
print("Classification Report for Test:")
print(classification_report(test_labels, test_preds, target_names=["background", "cargo", "tanker", "tug", "passengership"]))

# 10. 保存模型
torch.save(model.state_dict(), 'resnet18_model.pth')
