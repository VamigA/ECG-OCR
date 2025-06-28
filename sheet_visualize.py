import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from generator.ECGDataGenerator import ECGDataGenerator
from neural.sheet.ECGSheetDetectionModel import ECGSheetDetectionModel

generator = ECGDataGenerator()
image, data = generator.generate()
img = np.array(image)

plt.figure(figsize=(10, 10))
plt.imshow(img, cmap='gray')

sheet_corners = np.array(data['sheet'])
for i, (x, y) in enumerate(sheet_corners):
	plt.scatter([x], [y], color='lime', s=80, marker='o')
	plt.text(x, y - 10, f'True {i + 1}', color='lime', fontsize=12, weight='bold')

model = ECGSheetDetectionModel()
img_tensor = transforms.ToTensor()(image).unsqueeze(0)
state = torch.load('learning/model.pth', map_location='cpu')
if 'model' in state:
	model.load_state_dict(state['model'])
else:
	model.load_state_dict(state)

model.eval()
with torch.no_grad():
	pred_corners = model(img_tensor)[0].cpu().numpy().reshape(4, 2) * img.shape[0]
	for i, (x, y) in enumerate(pred_corners):
		plt.scatter([x], [y], color='red', s=80, marker='x')
		plt.text(x, y + 10, f'Pred {i + 1}', color='red', fontsize=12)

plt.title('ECG sheet corners: green = true, red = predicted', fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()