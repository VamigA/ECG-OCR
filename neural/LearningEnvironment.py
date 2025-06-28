from abc import ABC, abstractmethod
from accelerate import Accelerator
from datetime import datetime
import glob
import re
import torch
from tqdm import tqdm

class LearningEnvironment(ABC):
	def __init__(self, epochs=20, save_every=1, epoch_path='learning/epoch_{}.pth', model_path='learning/model.pth'):
		self.__epochs = epochs
		self.__save_every = save_every
		self.__epoch_path = epoch_path
		self.__model_path = model_path
		self.__accelerator = None

	@abstractmethod
	def setup(self):
		pass

	@abstractmethod
	def train_step(self, model, optimizer, batch):
		pass

	def on_epoch_start(self, model, optimizer, epoch):
		pass

	def on_epoch_end(self, model, optimizer, epoch, total_loss):
		pass

	def run(self):
		self.__accelerator = Accelerator()
		model, dataloader, optimizer = self.setup()

		checkpoint_files = glob.glob(self.__epoch_path.format('*'))
		if checkpoint_files:
			last_checkpoint = max(checkpoint_files, key=self.__extract_epoch)
			start_epoch = self.__load_model(model, optimizer, last_checkpoint)
			self.__accelerator.print(f'[{datetime.now()}][INFO] Loaded checkpoint from {last_checkpoint}.')
		else:
			start_epoch = 0

		model, dataloader, optimizer = self.__accelerator.prepare(model, dataloader, optimizer)
		model.train()

		for epoch in range(start_epoch, self.__epochs):
			total_loss = 0
			self.on_epoch_start(model, optimizer, epoch)

			bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{self.__epochs}', disable=not self.__accelerator.is_main_process)
			for batch in bar:
				loss = self.train_step(model, optimizer, batch)

				optimizer.zero_grad()
				self.__accelerator.backward(loss)
				optimizer.step()

				total_loss += loss.item()
				bar.set_postfix({'Loss': f'{loss.item():.6f}'})

			self.__accelerator.print(f'[{datetime.now()}][INFO] Epoch {epoch + 1} ended with total loss: {total_loss:.6f}!')
			self.on_epoch_end(model, optimizer, epoch, total_loss)

			if (epoch + 1) % self.__save_every == 0 or (epoch + 1) == self.__epochs:
				epoch_path = self.__epoch_path.format(epoch + 1)
				self.__save_model(epoch_path, model, optimizer, epoch + 1)
				self.__accelerator.print(f'[{datetime.now()}][INFO] Model saved at epoch {epoch + 1} as {epoch_path}.')

		self.__save_model(self.__model_path, model)
		self.__accelerator.print(f'[{datetime.now()}][INFO] Final model weights saved as {self.__model_path}.')

	def __save_model(self, path, model, optimizer=None, epoch=None):
		state_dict = self.__accelerator.unwrap_model(model).state_dict()
		if optimizer is not None and epoch is not None:
			self.__accelerator.save({
				'model': state_dict,
				'optimizer': optimizer.state_dict(),
				'epoch': epoch,
			}, path)
		else:
			self.__accelerator.save(state_dict, path)

	def __load_model(self, model, optimizer, path):
		checkpoint = torch.load(path, map_location='cpu')
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		return checkpoint.get('epoch', 0)

	@staticmethod
	def __extract_epoch(filename):
		match = re.search(r'(\d+)(?=\.pth$)', filename)
		return int(match.group(1)) if match else -1

__all__ = ('LearningEnvironment',)