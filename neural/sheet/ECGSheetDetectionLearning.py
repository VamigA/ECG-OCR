import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from neural.sheet.ECGSheetDetectionModel import ECGSheetDetectionModel
from neural.sheet.ECGSheetSyntheticDataset import ECGSheetSyntheticDataset
from neural.LearningEnvironment import LearningEnvironment

class ECGSheetDetectionLearning(LearningEnvironment):
	__BATCH_SIZE = 10
	__NUM_WORKERS = 2
	__LEARNING_RATE = 2e-5

	def setup(self):
		self.__mse_loss = nn.MSELoss()

		model = ECGSheetDetectionModel()
		dataloader = DataLoader(ECGSheetSyntheticDataset(), batch_size=self.__BATCH_SIZE,
			shuffle=True, num_workers=self.__NUM_WORKERS, pin_memory=True)
		optimizer = AdamW(model.parameters(), lr=self.__LEARNING_RATE)

		return (model, dataloader, optimizer)

	def train_step(self, model, _, batch):
		images, target = batch
		loss = self.__mse_loss(model(images), target)
		return loss

__all__ = ('ECGSheetDetectionLearning',)