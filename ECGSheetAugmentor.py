import albumentations as A
import cv2
import numpy as np
import os
from PIL import Image
import random

class ECGSheetAugmentor:
	__BACKGROUND_FOLDER = 'backgrounds'
	__IMG_WIDTH = 3360
	__IMG_HEIGHT = 3360

	def __init__(self) -> None:
		self.__backgrounds = []
		for filename in os.listdir(self.__BACKGROUND_FOLDER):
			path = os.path.join(self.__BACKGROUND_FOLDER, filename)
			image = cv2.imread(path)
			converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			resized = cv2.resize(converted, (self.__IMG_WIDTH, self.__IMG_HEIGHT))
			self.__backgrounds.append(resized)

		self.__augment = A.Compose([
			A.Rotate(limit=(-180, 180), border_mode=cv2.BORDER_REPLICATE, p=0.8),
			A.Perspective(scale=(0.02, 0.06), border_mode=cv2.BORDER_REPLICATE, p=0.6),
			A.RandomScale(scale_limit=0.05, p=0.4),
			A.OneOf([
				A.MotionBlur(blur_limit=3, p=0.3),
				A.Downscale(scale_range=(0.85, 0.95), interpolation_pair={
					'downscale': cv2.INTER_AREA,
					'upscale': cv2.INTER_LINEAR,
				}, p=0.5),
				A.ImageCompression(quality_range=(60, 100), p=0.5),
			], p=0.5),
			A.RandomShadow(p=0.3),
			A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
			A.GaussNoise(std_range=(0.05, 0.33), p=0.3),
			A.ISONoise(color_shift=(0.02, 0.15), intensity=(0.1, 0.3), p=0.2),
			A.MultiplicativeNoise(multiplier=(0.95, 1.05), per_channel=True, p=0.2),
		])

	def augment(self, pil_image: Image.Image) -> Image.Image:
		array = np.array(pil_image)
		image = self.__place_on_background(array)
		augmented = self.__augment(image=image)['image']
		return Image.fromarray(augmented)

	def __place_on_background(self, fg_image: np.ndarray) -> np.ndarray:
		scale = random.uniform(0.9, 1.1)
		fg_h, fg_w = fg_image.shape[:2]
		fg_w_new, fg_h_new = int(fg_w * scale), int(fg_h * scale)
		fg_resized = cv2.resize(fg_image, (fg_w_new, fg_h_new))

		w_diff = self.__IMG_WIDTH - fg_w_new
		h_diff = self.__IMG_HEIGHT - fg_h_new
		x_offset = int(random.gauss(w_diff // 2, w_diff // 8))
		y_offset = int(random.gauss(h_diff // 2, h_diff // 8))
		alpha = np.stack([fg_resized[:, :, 3] / 255] * 3, axis=2) * random.uniform(0.85, 1)

		result = random.choice(self.__backgrounds).copy()
		bg_crop = result[y_offset:y_offset + fg_h_new, x_offset:x_offset + fg_w_new]
		composite = (bg_crop * (1 - alpha) + fg_resized[:, :, :3] * alpha).astype(np.uint8)
		result[y_offset:y_offset + fg_h_new, x_offset:x_offset + fg_w_new] = composite
		return result

__all__ = ('ECGSheetAugmentor',)