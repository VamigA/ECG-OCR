import numpy as np
import torch
from torch.utils.data import Dataset

from generator.ECGDataGenerator import ECGDataGenerator
from generator.ECGLeadGenerator import ECGLeadGenerator

class ECGSyntheticDataset(Dataset):
	__DATASET_SIZE = 50000
	__IMAGE_SIZE = 1120
	__GRID_SIZE = 35
	__NUM_LEADS = 12

	def __init__(self):
		self.__generator = ECGDataGenerator()

	def __len__(self):
		return self.__DATASET_SIZE

	def __getitem__(self, _):
		image, signals, mm_per_sec = self.__generator.generate()
		image_array = np.array(image, dtype=np.float32)
		image_tensor = torch.tensor(image_array / 255).unsqueeze(0)

		target_bbox = torch.zeros((self.__NUM_LEADS, 4, self.__GRID_SIZE, self.__GRID_SIZE), dtype=torch.float32)
		target_presence = torch.zeros((self.__NUM_LEADS, self.__GRID_SIZE, self.__GRID_SIZE), dtype=torch.float32)

		for lead_index, lead_name in enumerate(ECGLeadGenerator.LEAD_NAMES):
			bbox = np.array(signals[lead_name]['bbox'], dtype=np.float32)
			if (bbox < 0).any():
				continue

			center_x = (bbox[0] + bbox[2]) / 2 / self.__IMAGE_SIZE
			center_y = (bbox[1] + bbox[3]) / 2 / self.__IMAGE_SIZE
			box_width = (bbox[2] - bbox[0]) / self.__IMAGE_SIZE
			box_height = (bbox[3] - bbox[1]) / self.__IMAGE_SIZE

			grid_x = min(int(center_x * self.__GRID_SIZE), self.__GRID_SIZE - 1)
			grid_y = min(int(center_y * self.__GRID_SIZE), self.__GRID_SIZE - 1)
			cell_x = center_x * self.__GRID_SIZE - grid_x
			cell_y = center_y * self.__GRID_SIZE - grid_y

			target_bbox[lead_index, :, grid_y, grid_x] = torch.tensor(
				[cell_x, cell_y, box_width, box_height], dtype=torch.float32)
			target_presence[lead_index, grid_y, grid_x] = 1

		mmps_label = 1 if mm_per_sec == 50 else 0
		mmps_tensor = torch.tensor(mmps_label, dtype=torch.float32)

		return (image_tensor, target_bbox, target_presence, mmps_tensor)

__all__ = ('ECGSyntheticDataset',)