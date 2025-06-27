import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from neural.detection.ECGDetectionModel import ECGDetectionModel
from neural.detection.ECGSyntheticDataset import ECGSyntheticDataset
from neural.LearningEnvironment import LearningEnvironment

class ECGDetectionLearning(LearningEnvironment):
	__BATCH_SIZE = 8
	__NUM_WORKERS = 2
	__LEARNING_RATE = 2e-5

	def setup(self):
		self.__mse_loss = nn.MSELoss()
		self.__bce_loss = nn.BCEWithLogitsLoss()

		model = ECGDetectionModel()
		dataloader = DataLoader(ECGSyntheticDataset(), batch_size=self.__BATCH_SIZE,
			shuffle=True, num_workers=self.__NUM_WORKERS, pin_memory=True)
		optimizer = AdamW(model.parameters(), lr=self.__LEARNING_RATE)

		return (model, dataloader, optimizer)

	def train_step(self, model, _, batch):
		images, target_bbox, mmps = batch
		pred_bbox, pred_mmps = model(images)

		bbox_loss = self.__mse_loss(pred_bbox, target_bbox)
		mmps_loss = self.__bce_loss(pred_mmps, mmps)
		loss = bbox_loss + mmps_loss
		return loss

__all__ = ('ECGDetectionLearning',)