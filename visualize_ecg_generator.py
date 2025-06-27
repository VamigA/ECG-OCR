import matplotlib.pyplot as plt
import numpy as np
import torch
from generator.ECGDataGenerator import ECGDataGenerator
from neural.detection.ECGDetectionModel import ECGDetectionModel
from torchvision import transforms

# Генерируем пример
generator = ECGDataGenerator()
image, signals, mm_per_sec = generator.generate()

# Преобразуем изображение в numpy
img = np.array(image)

# Вытаскиваем bbox из генератора
bboxes = [signals[name]['bbox'] for name in generator._ECGDataGenerator__lead_generator.LEAD_NAMES]
lead_names = generator._ECGDataGenerator__lead_generator.LEAD_NAMES

plt.figure(figsize=(10, 10))
plt.imshow(img, cmap='gray')

for bbox, name in zip(bboxes, lead_names):
    x0, y0, x1, y1 = bbox
    if min(bbox) < 0:
        continue  # отсутствует
    w, h = x1 - x0, y1 - y0
    rect = plt.Rectangle((x0, y0), w, h, linewidth=2, edgecolor='lime', facecolor='none')
    plt.gca().add_patch(rect)
    plt.text(x0, y0-5, name, color='lime', fontsize=10, weight='bold')

# === Предсказания модели ===
# Преобразуем изображение для модели (как в датасете)
transform = transforms.Compose([
    transforms.ToTensor(),  # (H, W) -> (1, H, W), float32 [0,1]
])
img_tensor = transform(image).unsqueeze(0)  # (1, 1, H, W)

# Загрузка модели
model = ECGDetectionModel()
state = torch.load('neural/detection/weights/model.pth', map_location='cpu')
if 'model' in state:
    model.load_state_dict(state['model'])
else:
    model.load_state_dict(state)
model.eval()

# Параметры сетки (должны совпадать с датасетом и моделью)
GRID_SIZE = 35
IMAGE_SIZE = img.shape[0]  # предполагается квадрат
NUM_LEADS = 12

with torch.no_grad():
    bbox_pred, presence_pred, mmps_pred = model(img_tensor)
    # bbox_pred: (1, 12, 4, GRID, GRID)
    # presence_pred: (1, 12, GRID, GRID)
    # Для каждого отведения ищем ячейку с максимальным presence
    for i, name in enumerate(lead_names):
        grid_idx = torch.argmax(presence_pred[0, i].reshape(-1))
        gy, gx = divmod(grid_idx.item(), GRID_SIZE)
        # bbox: (x, y, w, h) — x, y: offset внутри ячейки (0..1), w, h: относительные размеры
        cell = bbox_pred[0, i, :, gy, gx].cpu().numpy()
        cell_x, cell_y, box_w, box_h = cell
        # Центр bbox в относительных координатах изображения
        center_x = (gx + cell_x) / GRID_SIZE
        center_y = (gy + cell_y) / GRID_SIZE
        # Переводим в абсолютные координаты
        abs_w = box_w * IMAGE_SIZE
        abs_h = box_h * IMAGE_SIZE
        abs_x = int(center_x * IMAGE_SIZE - abs_w / 2)
        abs_y = int(center_y * IMAGE_SIZE - abs_h / 2)
        # Рисуем только если размеры разумные
        if box_w > 0.01 and box_h > 0.01:
            rect = plt.Rectangle((abs_x, abs_y), abs_w, abs_h, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
            plt.gca().add_patch(rect)
            plt.text(abs_x, abs_y + abs_h + 10, f'{name} pred', color='red', fontsize=9)
    # Предсказание mmps
    mmps_value = mmps_pred.item()
    mmps_label = 50 if mmps_value > 0.5 else 25
    plt.title(f'GT mm_per_sec: {mm_per_sec}, Predicted: {mmps_label} ({mmps_value:.2f})', fontsize=14)

plt.axis('off')
plt.tight_layout()
plt.show()
