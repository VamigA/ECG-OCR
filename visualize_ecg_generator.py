import matplotlib.pyplot as plt
from ECGDataGenerator import ECGDataGenerator
import numpy as np

# Генерируем пример
generator = ECGDataGenerator()
image, signals, _ = generator.generate()

# Преобразуем изображение в numpy
img = np.array(image)

# Вытаскиваем bbox
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

plt.axis('off')
plt.tight_layout()
plt.show()
