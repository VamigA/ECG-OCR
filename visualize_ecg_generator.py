import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from generator.ECGDataGenerator import ECGDataGenerator
from generator.ECGLeadGenerator import ECGLeadGenerator
from neural.detection.ECGDetectionModel import ECGDetectionModel

generator = ECGDataGenerator()
image, signals, mm_per_sec = generator.generate()
img = np.array(image)

plt.figure(figsize=(10, 10))
plt.imshow(img, cmap='gray')

for lead_name in ECGLeadGenerator.LEAD_NAMES:
	x0, y0, x1, y1 = signals[lead_name]['bbox']
	if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
		continue

	plt.gca().add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
		linewidth=2, edgecolor='lime', facecolor='none'))
	plt.text(x0, y0 - 5, lead_name, color='lime', fontsize=10, weight='bold')

model = ECGDetectionModel()
img_tensor = transforms.ToTensor()(image).unsqueeze(0)
state = torch.load('learning/epoch_1.pth', map_location='cpu')
if 'model' in state:
	model.load_state_dict(state['model'])
else:
	model.load_state_dict(state)

model.eval()
with torch.no_grad():
	bbox_pred, mmps_pred = model(img_tensor)
	for (cx, cy, w, h), name in zip(bbox_pred[0].cpu().numpy(), ECGLeadGenerator.LEAD_NAMES):
		if cx < 0 or cy < 0 or w < 0 or h < 0:
			continue

		abs_w, abs_h = w * img.shape[0], h * img.shape[0]
		abs_x, abs_y = int(cx * img.shape[0] - abs_w / 2), int(cy * img.shape[0] - abs_h / 2)

		plt.gca().add_patch(plt.Rectangle((abs_x, abs_y), abs_w, abs_h,
			linewidth=2, edgecolor='red', facecolor='none', linestyle='--'))
		plt.text(abs_x, abs_y + abs_h + 10, f'{name} pred', color='red', fontsize=9)

	mmps_value = mmps_pred.item()
	mmps_label = 50 if mmps_value > 0.5 else 25
	plt.title(f'mm_per_sec: {mm_per_sec}, predicted: {mmps_label} ({mmps_value:.2f})', fontsize=14)

plt.axis('off')
plt.tight_layout()
plt.show()