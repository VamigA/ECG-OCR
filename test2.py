import torch
import torch.nn as nn
from collections import OrderedDict
from ECGDataGenerator import ECGDataGenerator  # генератор тестовых данных
import matplotlib.pyplot as plt

# Параметры
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'ecg_cnn.pth'
TARGET_LENGTH = 800

# Модель
import torch.nn.functional as F
from torchvision import models

class ECGCNN(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(pretrained=True)

        # Заменим первый слой под ч/б:
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4  # ← (B, 512, H/32, W/32)
        )

        self.reduce = nn.Conv2d(512, 128, kernel_size=1)  # ↓ до 128 каналов
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=8),
            num_layers=4
        )
        self.head = nn.Linear(128, 12)

    def forward(self, x):  # (B, 1, 1120, 1120)
        x = self.backbone(x)       # (B, 512, 35, 35)
        x = self.reduce(x)         # (B, 128, 35, 35)
        x = x.flatten(2).permute(0, 2, 1)  # (B, 1225, 128)
        x = self.transformer(x)            # (B, 1225, 128)
        x = self.head(x)                   # (B, 1225, 12)
        x = x.permute(0, 2, 1)             # (B, 12, 1225)
        x = F.interpolate(x, size=800, mode='linear', align_corners=False)
        return x  # (B, 12, 800)

# Загрузи checkpoint
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

# Удали префикс 'module.' из имён ключей
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k.replace('module.', '') if k.startswith('module.') else k
    new_state_dict[new_key] = v

# 1. Создай модель и загрузи веса
model = ECGCNN().to(DEVICE)
model.load_state_dict(new_state_dict)
model.eval()

for _ in range(12):
    # 2. Сгенерируй 1 пример
    generator = ECGDataGenerator()
    image, target_signals = generator.generate()  # image — PIL.Image (ч/б), target_signals — dict

    # 3. Подготовь изображение
    import torchvision.transforms as T
    transform = T.Compose([
        T.ToTensor(),  # (1, H, W), значения [0, 1]
    ])
    x = transform(image).unsqueeze(0).to(DEVICE)  # (1, 1, 1120, 1120)

    plt.figure(figsize=(5, 5))
    plt.title("Input ECG Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

    # 4. Предсказание
    with torch.no_grad():
        y_pred = model(x)  # (1, 12, 800)
    pred = y_pred.squeeze(0).cpu().numpy()  # (12, 800)

    # 5. Визуализация
    fig, axes = plt.subplots(12, 1, figsize=(12, 16), sharex=True)
    for i, lead in enumerate(pred):
        axes[i].plot(lead, label=f'Lead {i+1}', color='blue')
        axes[i].legend(loc='upper right')
    plt.tight_layout()
    plt.show()