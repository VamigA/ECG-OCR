import numpy as np
import torch
from torch.utils.data import Dataset

from generator.ECGDataGenerator import ECGDataGenerator

class ECGSheetSyntheticDataset(Dataset):
	__DATASET_SIZE = 20000
	__IMAGE_SIZE = 1120

	def __init__(self):
		self.__generator = ECGDataGenerator()

	def __len__(self):
		return self.__DATASET_SIZE

	def __getitem__(self, _):
		image, data = self.__generator.generate_no_errors()
		image_array = np.array(image, dtype=np.float32) / 255
		image_tensor = torch.as_tensor(image_array).unsqueeze(0)

		corners = np.array(data['sheet'], dtype=np.float32) / self.__IMAGE_SIZE
		target = torch.from_numpy(corners).flatten()

		return (image_tensor, target)

__all__ = ('ECGSheetSyntheticDataset',)