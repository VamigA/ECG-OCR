from accelerate import Accelerator
import numpy as np
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
import traceback

from ECGDataGenerator import ECGDataGenerator
from ECGLeadGenerator import ECGLeadGenerator

IMAGE_SIZE = 1120
GRID_SIZE = 35
NUM_LEADS = 12

EPOCHS = 10
DATASET_SIZE = 50000
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
SAVE_EVERY = 5
MODEL_PATH = 'ecg_detection_model.pth'

class ECGSyntheticDataset(Dataset):
	def __init__(self):
		self.__generator = ECGDataGenerator()

	def __len__(self):
		return DATASET_SIZE

	def __getitem__(self, _):
		for _ in range(5):
			try:
				image, signals, mm_per_sec = self.__generator.generate()
				image_array = np.array(image, dtype=np.float32)
				image_tensor = torch.tensor(image_array / 255).unsqueeze(0)

				target_bbox = torch.zeros((NUM_LEADS, 4, GRID_SIZE, GRID_SIZE), dtype=torch.float32)
				target_presence = torch.zeros((NUM_LEADS, GRID_SIZE, GRID_SIZE), dtype=torch.float32)

				for lead_index, lead_name in enumerate(ECGLeadGenerator.LEAD_NAMES):
					bbox = np.array(signals[lead_name]['bbox'], dtype=np.float32)
					if (bbox < 0).any():
						continue

					center_x = (bbox[0] + bbox[2]) / 2 / IMAGE_SIZE
					center_y = (bbox[1] + bbox[3]) / 2 / IMAGE_SIZE
					box_width = (bbox[2] - bbox[0]) / IMAGE_SIZE
					box_height = (bbox[3] - bbox[1]) / IMAGE_SIZE

					grid_x = min(int(center_x * GRID_SIZE), GRID_SIZE - 1)
					grid_y = min(int(center_y * GRID_SIZE), GRID_SIZE - 1)
					cell_x = center_x * GRID_SIZE - grid_x
					cell_y = center_y * GRID_SIZE - grid_y

					target_bbox[lead_index, :, grid_y, grid_x] = torch.tensor(
						[cell_x, cell_y, box_width, box_height], dtype=torch.float32)
					target_presence[lead_index, grid_y, grid_x] = 1

				mmps_label = 1 if mm_per_sec == 50 else 0
				mmps_tensor = torch.tensor(mmps_label, dtype=torch.float32)

				return (image_tensor, target_bbox, target_presence, mmps_tensor)
			except Exception as exception:
				with open('errors.log', 'a') as log_file:
					log_file.write(f'[ERROR] {type(exception).__name__}: {exception}\n')
					log_file.write(traceback.format_exc())

				continue

		raise

class ECGDetectionModel(nn.Module):
	def __init__(self):
		super().__init__()

		self.backbone = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
			*list(resnet18(weights=ResNet18_Weights.DEFAULT).children())[1:-2],
		)

		self.detection_head = nn.Conv2d(512, NUM_LEADS * 5, kernel_size=1)

		self.mmps_head = nn.Sequential(
			nn.AdaptiveAvgPool2d((1, 1)),
			nn.Flatten(),
			nn.Linear(512, 1),
		)

	def forward(self, images):
		features = self.backbone(images) # (BATCH_SIZE, 512, GRID_SIZE, GRID_SIZE)
		detection = self.detection_head(features) # (BATCH_SIZE, NUM_LEADS*5, GRID_SIZE, GRID_SIZE)
		detection = detection.view(BATCH_SIZE, NUM_LEADS, 5, GRID_SIZE, GRID_SIZE)

		bbox = torch.sigmoid(detection[:, :, :4, :, :]) # (BATCH_SIZE, NUM_LEADS, 4, GRID_SIZE, GRID_SIZE)
		presence = detection[:, :, 4, :, :] # (BATCH_SIZE, NUM_LEADS, GRID_SIZE, GRID_SIZE)
		mmps = self.mmps_head(features) # (BATCH_SIZE, 1)
		return (bbox, presence, mmps)

def save_model(model, optimizer, epoch, path):
	torch.save({
		'model': model.state_dict(),
		'optimizer': optimizer.state_dict(),
		'epoch': epoch,
	}, path)

def load_model(model, optimizer, path):
	checkpoint = torch.load(path, map_location='cpu')
	model.load_state_dict(checkpoint['model'])
	optimizer.load_state_dict(checkpoint['optimizer'])
	return checkpoint.get('epoch', 0)

def main():
	accelerator = Accelerator()
	dataset = ECGSyntheticDataset()
	model = ECGDetectionModel()
	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
	optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
	bce_loss = nn.BCEWithLogitsLoss()

	if os.path.exists(MODEL_PATH):
		start_epoch = load_model(model, optimizer, MODEL_PATH)
		print(f'[INFO] Loaded checkpoint from epoch {start_epoch}.')
	else:
		start_epoch = 0

	model, dataloader, optimizer = accelerator.prepare(model, dataloader, optimizer)
	model.train()

	for epoch in range(start_epoch, EPOCHS):
		total_loss = total_bbox_loss = total_presence_loss = total_mmps_loss = 0
		bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{EPOCHS}:', disable=not accelerator.is_main_process)

		for images, target_bbox, target_presence, mmps in bar:
			mask = target_presence > 0.5
			pred_bbox, pred_presence, pred_mmps = model(images)

			bbox_loss = (((pred_bbox - target_bbox) ** 2).sum(2) * mask).sum() / mask.sum().clamp(min=1)
			presence_loss = bce_loss(pred_presence, target_presence)
			mmps_loss = bce_loss(pred_mmps, mmps.unsqueeze(1))
			loss = bbox_loss + presence_loss + mmps_loss

			optimizer.zero_grad()
			accelerator.backward(loss)
			optimizer.step()

			total_loss += loss.item()
			total_bbox_loss += bbox_loss.item()
			total_presence_loss += presence_loss.item()
			total_mmps_loss += mmps_loss.item()

			bar.set_postfix({
				'Loss:': f'{loss.item():.4f}',
				'1:': f'{bbox_loss.item():.4f}',
				'2:': f'{presence_loss.item():.4f}',
				'3:': f'{mmps_loss.item():.4f}',
			})

		n_batches = len(dataloader)
		avg_loss = total_loss / n_batches
		avg_bbox = total_bbox_loss / n_batches
		avg_pres = total_presence_loss / n_batches
		avg_mmps = total_mmps_loss / n_batches
		accelerator.print(f'Epoch {epoch + 1}: loss = {avg_loss:.3f} ({avg_bbox:.3f}, {avg_pres:.3f}, {avg_mmps:.3f})')

		if (epoch + 1) % SAVE_EVERY == 0 or (epoch + 1) == EPOCHS:
			if accelerator.is_main_process:
				save_model(model, optimizer, epoch + 1, MODEL_PATH)
				print(f'[INFO] Model saved at epoch {epoch + 1}.')

if __name__ == '__main__':
	main()