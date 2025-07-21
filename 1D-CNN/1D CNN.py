import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 1. 设定数据集路径
DATASET_PATH = r"/root/autodl-tmp/inclusion_3000_exclusion_5000"

# 2. 定义类别
CLASSES = ["background", "cargo", "passengership", "tanker", "tug"]
NUM_CLASSES = len(CLASSES)

# 3. 加载音频数据
class ShipNoiseDataset(Dataset):
    def __init__(self, root_dir, target_length=16000):
        self.file_list = []
        self.labels = []
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
        waveform = torch.nan_to_num(waveform)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if sample_rate != 16000:
            waveform = T.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        if waveform.shape[1] < self.target_length:
            waveform = torch.nn.functional.pad(waveform, (0, self.target_length - waveform.shape[1]))
        else:
            waveform = waveform[:, :self.target_length]
        
        return waveform, label

# 4. 构造数据集和 DataLoader
train_loader = DataLoader(ShipNoiseDataset(os.path.join(DATASET_PATH, "train")), batch_size=32, shuffle=True)
val_loader = DataLoader(ShipNoiseDataset(os.path.join(DATASET_PATH, "validation")), batch_size=32, shuffle=False)
test_loader = DataLoader(ShipNoiseDataset(os.path.join(DATASET_PATH, "test")), batch_size=32, shuffle=False)

# 5. 定义 1D CNN 模型
class Ship1DCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(Ship1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=9, stride=3)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=6, stride=1)
        self.pool2 = nn.AvgPool1d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=6, stride=1)

        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 16000)
            out = self.conv1(dummy_input)
            out = self.pool1(out)
            out = self.conv2(out)
            out = self.pool2(out)
            out = self.conv3(out)
            self._to_linear = out.view(1, -1).size(1)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 6. 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Ship1DCNN(num_classes=NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20
train_losses, val_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for waveforms, labels in train_loader:
        waveforms, labels = waveforms.to(device), labels.to(device)
        waveforms = waveforms.view(waveforms.size(0), 1, -1)
        optimizer.zero_grad()
        loss = criterion(model(waveforms), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for waveforms, labels in val_loader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            waveforms = waveforms.view(waveforms.size(0), 1, -1)
            outputs = model(waveforms)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    val_accuracies.append(correct / total)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, Val Acc: {val_accuracies[-1] * 100:.2f}%")

# 7. 绘制 Loss 和 Accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# 8. 评估模型
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for waveforms, labels in loader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            waveforms = waveforms.view(waveforms.size(0), 1, -1)
            outputs = model(waveforms)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

print(f"Test Accuracy: {evaluate(model, test_loader) * 100:.2f}%")
