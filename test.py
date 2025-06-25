# ecg_training.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
import numpy as np
from tqdm import tqdm

from ECGDataGenerator import ECGDataGenerator

# Константы
BATCH_SIZE = 25
EPOCHS = 50
LEARNING_RATE = 1e-6
TARGET_LENGTH = 800
IMAGE_SIZE = (1120, 1120)

# Случайный генератор для отладки
class SyntheticECGDataset(Dataset):
    def __init__(self, count):
        self.generator = ECGDataGenerator()
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        image, signals = self.generator.generate()
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.tensor(image).unsqueeze(0)  # (1, H, W)

        targets = [torch.tensor(signals[name], dtype=torch.float32) for name in self.generator._ECGDataGenerator__lead_generator.LEAD_NAMES]
        target = torch.stack(targets, dim=0)  # (12, 800)

        return image, target

# Модель
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class ECGCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Используем ResNet18, адаптированный под ч/б вход
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4  # (B, 512, 35, 35)
        )

        # Сжимаем количество каналов до 256
        self.reduce = nn.Conv2d(512, 256, kernel_size=1)
        self.norm = nn.LayerNorm([256, 35, 35])
        self.dropout = nn.Dropout(0.1)

        # Двумерное позиционное кодирование (learnable)
        self.row_embed = nn.Parameter(torch.zeros(1, 35, 128))  # (1, 35, 128)
        self.col_embed = nn.Parameter(torch.zeros(1, 35, 128))  # (1, 35, 128)
        nn.init.trunc_normal_(self.row_embed, std=0.02)
        nn.init.trunc_normal_(self.col_embed, std=0.02)

        # Transformer для обработки временной оси (ширины)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=16, dim_feedforward=1024, dropout=0.2, batch_first=True),
            num_layers=8
        )

        # Линейная голова для предсказания 12 каналов
        self.head = nn.Linear(256, 12)

    def forward(self, x):
        x = self.backbone(x)              # (B, 512, 35, 35)
        x = self.reduce(x)                # (B, 256, 35, 35)
        x = self.norm(x)                  # (B, 256, 35, 35)
        x = self.dropout(x)               # (B, 256, 35, 35)
        x = x.flatten(2).permute(0, 2, 1)

        x += torch.cat([
            self.row_embed[:, :35, :].unsqueeze(2).expand(-1, 35, 35, -1),
            self.col_embed[:, :35, :].unsqueeze(1).expand(-1, 35, 35, -1),
        ], dim=-1).reshape(1, 1225, 256)

        x = self.transformer(x)            # (B, 1225, 256)
        x = x.permute(0, 2, 1)             # (B, 256, 1225)
        x = F.interpolate(x, size=800, mode='linear', align_corners=False)  # (B, 256, 800)
        x = self.head(x.permute(0, 2, 1)).permute(0, 2, 1)                  # (B, 12, 800)
        return x  # (B, 12, 800)

# Тренировка
accelerator = Accelerator()

model = ECGCNN()
dataset = SyntheticECGDataset(count=2500)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

model, dataloader, optimizer = accelerator.prepare(model, dataloader, optimizer)

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for images, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        preds = model(images)
        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    accelerator.print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")

# Сохранение модели
if accelerator.is_main_process:
    torch.save(model.state_dict(), "ecg_cnn.pth")
    print("\nModel saved as ecg_cnn.pth")