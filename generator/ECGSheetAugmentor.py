import albumentations as A
import cv2
import numpy as np
import os
from PIL import Image
import random

class ECGSheetAugmentor:
	__BACKGROUND_FOLDER = 'generator/backgrounds'
	__IMG_WIDTH = 1120
	__IMG_HEIGHT = 1120

	def __init__(self) -> None:
		self.__backgrounds = []
		for filename in os.listdir(self.__BACKGROUND_FOLDER):
			path = os.path.join(self.__BACKGROUND_FOLDER, filename)
			image = cv2.imread(path)
			converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			resized = cv2.resize(converted, (self.__IMG_WIDTH, self.__IMG_HEIGHT))
			self.__backgrounds.append(resized)

		self.__augment = A.Compose(
			transforms=[
				A.Rotate(limit=(-180, 180), border_mode=cv2.BORDER_REPLICATE, p=0.8),
				A.Perspective(scale=(0.02, 0.06), border_mode=cv2.BORDER_REPLICATE, p=0.6),
				A.RandomScale(scale_limit=0.05, p=0.4),
				A.CenterCrop(
					height=self.__IMG_HEIGHT, width=self.__IMG_WIDTH,
					pad_if_needed=True, border_mode=cv2.BORDER_REPLICATE,
				),
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
				A.GaussNoise(std_range=(0.05, 0.15), p=0.3),
				A.ISONoise(color_shift=(0.02, 0.07), intensity=(0.1, 0.3), p=0.2),
				A.MultiplicativeNoise(multiplier=(0.95, 1.05), per_channel=True, p=0.2),
			],
			keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
		)

	def augment(
		self, pil_image: Image.Image, layout: dict[str, tuple[int, int, int, int]],
	) -> tuple[Image.Image, dict[str, tuple[tuple[int, int], ...]]]:
		w, h = pil_image.width, pil_image.height
		new_layout = {'sheet': ((0, 0), (w, 0), (w, h), (0, h))}
		for key, (x0, y0, x1, y1) in layout.items():
			new_layout[key] = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))

		image = self.__place_on_background(np.array(pil_image), new_layout)

		augmented = self.__augment(
			image=image,
			keypoints=tuple(point for key in new_layout.keys() for point in new_layout[key]),
			keypoint_labels=('',) * len(new_layout) * 4,
		)

		for i, key in enumerate(new_layout.keys()):
			new_layout[key] = tuple((int(x), int(y)) for x, y in augmented['keypoints'][i * 4:(i + 1) * 4])

		return (Image.fromarray(augmented['image']), new_layout)

	def __place_on_background(self, fg_image: np.ndarray, layout: dict[str, tuple[tuple[int, int], ...]]) -> np.ndarray:
		scale = random.uniform(0.9, 1.1)
		fg_h, fg_w = fg_image.shape[:2]
		fg_w_new, fg_h_new = int(fg_w * scale), int(fg_h * scale)
		fg_resized = cv2.resize(fg_image, (fg_w_new, fg_h_new))

		w_diff = self.__IMG_WIDTH - fg_w_new
		h_diff = self.__IMG_HEIGHT - fg_h_new
		x_offset = np.clip(int(random.gauss(w_diff // 2, w_diff // 8)), 0, w_diff)
		y_offset = np.clip(int(random.gauss(h_diff // 2, h_diff // 8)), 0, h_diff)
		alpha = np.stack([fg_resized[:, :, 3] / 255] * 3, axis=2) * random.uniform(0.85, 1)

		result = random.choice(self.__backgrounds).copy()
		bg_crop = result[y_offset:y_offset + fg_h_new, x_offset:x_offset + fg_w_new]
		composite = (bg_crop * (1 - alpha) + fg_resized[:, :, :3] * alpha).astype(np.uint8)
		result[y_offset:y_offset + fg_h_new, x_offset:x_offset + fg_w_new] = composite

		for key, points in layout.items():
			layout[key] = tuple((int(x * scale) + x_offset, int(y * scale) + y_offset) for (x, y) in points)

		return result

__all__ = ('ECGSheetAugmentor',)