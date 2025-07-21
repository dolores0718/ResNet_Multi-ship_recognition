import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

# 1. 设定数据集路径
DATASET_PATH = r"E:\毕业设计\船舶辐射噪声\inclusion_3000_exclusion_5000"

# 2. 定义类别
CLASSES = ["background", "cargo", "passengership", "tanker", "tug"]
NUM_CLASSES = len(CLASSES)


# 3. 加载音频数据
class ShipNoiseDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_length=16000):
        self.file_list = []
        self.labels = []
        self.transform = transform
        self.target_length = target_length

        for label, category in enumerate(CLASSES):
            category_path = os.path.join(root_dir, "audio", category)
            for file in os.listdir(category_path):
                if file.endswith(".wav"):
                    self.file_list.append(os.path.join(category_path, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_path = self.file_list[idx]
        label = self.labels[idx]

        waveform, sample_rate = torchaudio.load(audio_path)

        # 处理 NaN 和 Inf
        waveform = torch.nan_to_num(waveform)

        # 保证音频为单通道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 统一采样率
        if sample_rate != 16000:
            resample = T.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resample(waveform)

        # 归一化
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val

        # 处理长度，填充或截断
        if waveform.shape[1] < self.target_length:
            pad = torch.zeros(1, self.target_length)
            pad[:, :waveform.shape[1]] = waveform
            waveform = pad
        else:
            waveform = waveform[:, :self.target_length]

        return waveform, label


# 4. 构造数据集和 DataLoader
train_dataset = ShipNoiseDataset(os.path.join(DATASET_PATH, "train"))
val_dataset = ShipNoiseDataset(os.path.join(DATASET_PATH, "validation"))
test_dataset = ShipNoiseDataset(os.path.join(DATASET_PATH, "test"))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 5. 定义 1D CNN 模型
class Ship1DCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(Ship1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=9, stride=3)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=6, stride=1)
        self.pool2 = nn.AvgPool1d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=6, stride=1)

        # 计算flatten的维度
        self._to_linear = None
        self._compute_flatten_dim()

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def _compute_flatten_dim(self):
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 16000)  # 确保输入大小一致
            out = self.conv1(dummy_input)
            out = self.pool1(out)
            out = self.conv2(out)
            out = self.pool2(out)
            out = self.conv3(out)
            self._to_linear = out.view(1, -1).size(1)  # 计算展平后的维度

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


# 6. 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Ship1DCNN(num_classes=NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        waveforms, labels = batch
        waveforms, labels = waveforms.to(device), labels.to(device)

        # 改变 waveforms 维度: [batch, 1, 16000]
        waveforms = waveforms.view(waveforms.size(0), 1, -1)

        optimizer.zero_grad()
        outputs = model(waveforms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")


# 7. 评估模型
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            waveforms, labels = batch
            waveforms, labels = waveforms.to(device), labels.to(device)

            waveforms = waveforms.view(waveforms.size(0), 1, -1)  # 统一格式
            outputs = model(waveforms)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total


accuracy = evaluate(model, test_loader)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
