import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from neural.detection.ECGDetectionModel import ECGDetectionModel
from neural.detection.ECGSyntheticDataset import ECGSyntheticDataset
from neural.LearningEnvironment import LearningEnvironment

class ECGDetectionLearning(LearningEnvironment):
	__BATCH_SIZE = 32
	__NUM_WORKERS = 2
	__LEARNING_RATE = 2e-5

	def setup(self):
		self.__bce_loss = nn.BCEWithLogitsLoss()

		model = ECGDetectionModel()
		dataloader = DataLoader(ECGSyntheticDataset(), batch_size=self.__BATCH_SIZE,
			shuffle=True, num_workers=self.__NUM_WORKERS, pin_memory=True)
		optimizer = AdamW(model.parameters(), lr=self.__LEARNING_RATE)

		return (model, dataloader, optimizer)

	def train_step(self, model, _, batch):
		images, target_bbox, target_presence, mmps = batch
		pred_bbox, pred_presence, pred_mmps = model(images)

		mask = target_presence > 0.5
		bbox_loss = (((pred_bbox - target_bbox) ** 2).sum(2) * mask).sum() / mask.sum().clamp(min=1)
		presence_loss = self.__bce_loss(pred_presence, target_presence)
		mmps_loss = self.__bce_loss(pred_mmps, mmps.unsqueeze(1))

		loss = bbox_loss + presence_loss + mmps_loss
		return loss

__all__ = ('ECGDetectionLearning',)