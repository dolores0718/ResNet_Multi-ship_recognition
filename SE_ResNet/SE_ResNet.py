import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torchvision.models import resnet18

# 1. 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_path, label_path, transform=None):
        data = np.load(data_path)
        self.images = data['data']
        self.labels = data['labels']

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

# 2. 定义 SE 模块
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),  # 压缩
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),  # 激励
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        # 全局平均池化
        y = self.global_avg_pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1, 1)
        # 将比例权重乘到原始输入特征图上
        return x * y.expand_as(x)

# 3. 定义带有 SE 模块的 BasicBlock
class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction)

        # 下采样层（用于匹配输入输出的尺寸）
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 插入 SE 模块
        out = self.se(out)

        out += identity
        out = self.relu(out)

        return out

# 4. 定义 SE_ResNet18
class SE_ResNet(nn.Module):
    def __init__(self, num_classes=5):
        super(SE_ResNet, self).__init__()
        base_model = resnet18(pretrained=True)
        self.in_channels = 64

        # 替换最初的 conv1 和 bn1
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        # 定义 ResNet 的结构（用 SEBasicBlock 替换原来的 BasicBlock）
        self.layer1 = self._make_layer(SEBasicBlock, 64, 2)
        self.layer2 = self._make_layer(SEBasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(SEBasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(SEBasicBlock, 512, 2, stride=2)

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# 5. 定义数据增强和预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 6. 加载数据集
train_dataset = CustomDataset('/root/autodl-tmp/encoded_data/train_encoded.npz', 
                              '/root/autodl-tmp/encoded_data/train_labels.npz', 
                              transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 7. 定义模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SE_ResNet(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 8. 训练模型
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

# 9. 保存模型
torch.save(model.state_dict(), 'se_resnet18_model.pth')
print("Model saved as 'se_resnet18_model.pth'")
