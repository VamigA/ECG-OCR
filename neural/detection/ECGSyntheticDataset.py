import numpy as np
import torch
from torch.utils.data import Dataset

from generator.ECGDataGenerator import ECGDataGenerator
from generator.ECGLeadGenerator import ECGLeadGenerator

class ECGSyntheticDataset(Dataset):
	__DATASET_SIZE = 50000
	__IMAGE_SIZE = 1120

	def __init__(self):
		self.__generator = ECGDataGenerator()

	def __len__(self):
		return self.__DATASET_SIZE

	def __getitem__(self, _):
		image, signals, mm_per_sec = self.__generator.generate()
		image_tensor = torch.as_tensor(np.array(image, dtype=np.float32) / 255).unsqueeze(0)
		target_bbox = torch.full((12, 4), -1, dtype=torch.float32)

		for i, name in enumerate(ECGLeadGenerator.LEAD_NAMES):
			x0, y0, x1, y1 = signals[name]['bbox']
			if min(x0, y0, x1, y1) < 0:
				continue

			cx = (x0 + x1) / 2 / self.__IMAGE_SIZE
			cy = (y0 + y1) / 2 / self.__IMAGE_SIZE
			w = (x1 - x0) / self.__IMAGE_SIZE
			h = (y1 - y0) / self.__IMAGE_SIZE

			target_bbox[i] = torch.tensor([cx, cy, w, h], dtype=torch.float32)

		mmps_tensor = torch.tensor(mm_per_sec == 50, dtype=torch.float32)
		return (image_tensor, target_bbox, mmps_tensor)

__all__ = ('ECGSyntheticDataset',)