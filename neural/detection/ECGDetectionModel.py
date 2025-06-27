import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ECGDetectionModel(nn.Module):
	__GRID_SIZE = 35
	__NUM_LEADS = 12

	def __init__(self):
		super().__init__()

		self.backbone = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
			*list(resnet18(weights=ResNet18_Weights.DEFAULT).children())[1:-2],
		)

		self.detection_head = nn.Conv2d(512, self.__NUM_LEADS * 5, kernel_size=1)

		self.mmps_head = nn.Sequential(
			nn.AdaptiveAvgPool2d((1, 1)),
			nn.Flatten(),
			nn.Linear(512, 1),
		)

	def forward(self, images):
		batch_size = images.size(0)
		features = self.backbone(images)
		detection = self.detection_head(features)
		detection = detection.view(batch_size, self.__NUM_LEADS, 5, self.__GRID_SIZE, self.__GRID_SIZE)

		bbox = torch.sigmoid(detection[:, :, :4, :, :])
		presence = detection[:, :, 4, :, :]
		mmps = self.mmps_head(features)
		return (bbox, presence, mmps)

__all__ = ('ECGDetectionModel',)