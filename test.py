import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
import numpy as np
import os
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights
import traceback

from ECGDataGenerator import ECGDataGenerator

# Константы
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 2e-5
SAVE_EVERY = 5
MODEL_PATH = 'ecg_box_presence.pth'

# Случайный генератор для отладки
class SyntheticECGBoxDataset(Dataset):
	def __init__(self, count):
		self.generator = ECGDataGenerator()
		self.count = count
		self.lead_names = self.generator._ECGDataGenerator__lead_generator.LEAD_NAMES

	def __len__(self):
		return self.count

	def __getitem__(self, _):
		for _ in range(5):
			try:
				image, signals, mm_per_sec = self.generator.generate()
				image = torch.tensor(np.array(image, dtype=np.float32) / 255).unsqueeze(0)  # (1, H, W)
				bboxes = [torch.tensor(signals[name]['bbox'], dtype=torch.float32) for name in self.lead_names]
				presence = [1 if (bbox > 0).all() else 0 for bbox in bboxes]
				bboxes = torch.stack(bboxes, dim=0)  # (12, 4)
				presence = torch.tensor(presence, dtype=torch.float32)  # (12,)
				mmps = 1 if mm_per_sec == 50 else 0
				mmps = torch.tensor(mmps, dtype=torch.float32)
				return (image, bboxes, presence, mmps)
			except Exception as e:
				with open('ecg_data_gen_errors.log', 'a') as log:
					log.write(f'[ERROR] {type(e).__name__}: {e}\n')
					log.write(traceback.format_exc())

				continue

		raise

# Модель
class ECGBoxPresenceRegressor(nn.Module):
	def __init__(self):
		super().__init__()
		base = resnet18(weights=ResNet18_Weights.DEFAULT)
		base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.backbone = nn.Sequential(*list(base.children())[:-2])  # (B, 512, 35, 35)
		self.pool = nn.AdaptiveAvgPool2d((1, 1))
		self.bn = nn.BatchNorm1d(512)
		self.dropout = nn.Dropout(0.2)
		self.bbox_head = nn.Linear(512, 48)
		self.presence_head = nn.Linear(512, 12)
		self.mmps_head = nn.Linear(512, 1)

	def forward(self, x):
		x = self.backbone(x)
		x = self.pool(x).flatten(1)
		x = self.bn(x)
		x = self.dropout(x)
		bbox = self.bbox_head(x).view(-1, 12, 4)
		presence = self.presence_head(x)
		mmps = self.mmps_head(x)
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

# Тренировка
def main():
	accelerator = Accelerator()
	model = ECGBoxPresenceRegressor()
	dataset = SyntheticECGBoxDataset(count=10000)
	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
	optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
	start_epoch = 0

	if os.path.exists(MODEL_PATH):
		start_epoch = load_model(model, optimizer, MODEL_PATH)
		print(f'[INFO] Loaded checkpoint from epoch {start_epoch}')

	model, dataloader, optimizer = accelerator.prepare(model, dataloader, optimizer)
	model.train()

	for epoch in range(start_epoch, EPOCHS):
		total_loss, total_bbox_loss, total_presence_loss, total_mmps_loss = 0, 0, 0, 0
		n_batches = len(dataloader)
		bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}')
		for images, bboxes, presence, mmps in bar:
			pred_bboxes, pred_presence, pred_mmps = model(images)
			mask = (presence > 0.5)
			bbox_loss = (((pred_bboxes - bboxes) ** 2).sum(-1) * mask).sum() / mask.sum().clamp(min=1)
			presence_loss = nn.BCEWithLogitsLoss()(pred_presence, presence)
			mmps_loss = nn.BCEWithLogitsLoss()(pred_mmps, mmps.unsqueeze(1))
			loss = bbox_loss + presence_loss + mmps_loss

			optimizer.zero_grad()
			accelerator.backward(loss)
			optimizer.step()

			total_loss += loss.item()
			total_bbox_loss += bbox_loss.item()
			total_presence_loss += presence_loss.item()
			total_mmps_loss += mmps_loss.item()
			bar.set_postfix({
				'loss': f'{loss.item():.4f}',
				'bbox': f'{bbox_loss.item():.4f}',
				'pres': f'{presence_loss.item():.4f}',
				'mmps': f'{mmps_loss.item():.4f}',
			})

		avg_loss = total_loss / n_batches
		avg_bbox = total_bbox_loss / n_batches
		avg_pres = total_presence_loss / n_batches
		avg_mmps = total_mmps_loss / n_batches
		accelerator.print(f'Epoch {epoch+1}: Loss={avg_loss:.6f} | BBox={avg_bbox:.6f} | Presence={avg_pres:.6f} | MMps={avg_mmps:.6f}')

		if (epoch + 1) % SAVE_EVERY == 0 or (epoch + 1) == EPOCHS:
			if accelerator.is_main_process:
				save_model(model, optimizer, epoch + 1, MODEL_PATH)
				print(f'[INFO] Model saved at epoch {epoch+1}')

if __name__ == '__main__':
	main()