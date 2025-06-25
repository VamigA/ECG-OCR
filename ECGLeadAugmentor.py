import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random

class ECGLeadAugmentor:
	__rotation_probability = 0.4
	__perspective_probability = 0.4
	__wave_probability = 0.2
	__vignette_probability = 0.1
	__noise_probability = 0.4
	__blur_probability = 0.3
	__brightness_probability = 0.5
	__contrast_probability = 0.5
	__color_shift_probability = 0.3
	__color_invert_probability = 0.05

	def augment(self, img: Image.Image) -> Image.Image:
		img = img.copy()

		if random.random() <= self.__rotation_probability:
			img = self.__apply_rotation(img)
		if random.random() <= self.__perspective_probability:
			img = self.__apply_perspective(img)
		if random.random() <= self.__wave_probability:
			img = self.__apply_wave(img)
		if random.random() <= self.__vignette_probability:
			img = self.__apply_vignette(img)
		if random.random() <= self.__noise_probability:
			img = self.__add_noise(img)
		if random.random() <= self.__blur_probability:
			img = self.__add_blur(img)
		if random.random() <= self.__brightness_probability:
			img = self.__adjust_brightness(img)
		if random.random() <= self.__contrast_probability:
			img = self.__adjust_contrast(img)
		if random.random() <= self.__color_shift_probability:
			img = self.__shift_color(img)
		if random.random() <= self.__color_invert_probability:
			img = self.__invert_color(img)

		return img

	def __apply_rotation(self, img: Image.Image) -> Image.Image:
		angle = random.uniform(-5, 5)
		return img.rotate(angle, Image.BICUBIC)

	def __apply_perspective(self, img: Image.Image) -> Image.Image:
		w, h = img.size

		coefficients = self.__find_coefficients(
			[
				(random.randint(-20, 20), random.randint(-20, 20)),
				(w + random.randint(-20, 20), random.randint(-20, 20)),
				(w + random.randint(-20, 20), h + random.randint(-20, 20)),
				(random.randint(-20, 20), h + random.randint(-20, 20)),
			],
			[(0, 0), (w, 0), (w, h), (0, h)],
		)

		return img.transform(img.size, Image.PERSPECTIVE, coefficients, Image.BICUBIC)

	def __find_coefficients(self, pa: list[tuple[float, float]], pb: list[tuple[float, float]]) -> np.ndarray:
		matrix = []
		for p1, p2 in zip(pa, pb):
			matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
			matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

		A = np.matrix(matrix, dtype=np.float64)
		B = np.array(pb).reshape(8)
		res = np.dot(np.linalg.pinv(A), B)
		return res.A1

	def __apply_wave(self, img: Image.Image) -> Image.Image:
		arr = np.array(img)
		for axis in (0, 1):
			if random.choice([False, True]):
				arr = self.__wave_distortion(arr, axis)

		return Image.fromarray(arr)

	def __wave_distortion(self, arr: np.ndarray, axis: int) -> np.ndarray:
		frequency = random.uniform(2, 5) / arr.shape[axis]
		amplitude = random.uniform(2, 5)

		result = np.zeros_like(arr)
		for i in range(arr.shape[axis]):
			shift = int(amplitude * np.sin(2 * np.pi * frequency * i))
			if axis == 0:
				result[i] = np.roll(arr[i], shift, 0)
			else:
				result[:, i] = np.roll(arr[:, i], shift, 0)

		return result

	def __apply_vignette(self, img: Image.Image) -> Image.Image:
		w, h = img.size
		x = np.linspace(random.uniform(-2, -0.1), random.uniform(0.1, 2), w)
		y = np.linspace(random.uniform(-2, -0.1), random.uniform(0.1, 2), h)
		xv, yv = np.meshgrid(x, y)

		mask = np.clip(1 - (xv ** 2 + yv ** 2), 0.5, 1)[..., None]

		arr = np.array(img).astype(np.float32)
		arr[..., :3] = np.clip(arr[..., :3] * mask, 0, 255)
		return Image.fromarray(arr.astype(np.uint8))

	def __add_noise(self, img: Image.Image) -> Image.Image:
		arr = np.array(img).astype(np.int16)
		noise = np.random.normal(0, 50, arr.shape).astype(np.int16)
		res = np.clip(arr + noise, 0, 255).astype(np.uint8)
		return Image.fromarray(res)

	def __add_blur(self, img: Image.Image) -> Image.Image:
		radius = random.uniform(0.2, 0.6)
		blur_filter = ImageFilter.GaussianBlur(radius)
		return img.filter(blur_filter)

	def __adjust_brightness(self, img: Image.Image) -> Image.Image:
		factor = random.uniform(0.7, 1.3)
		enhancer = ImageEnhance.Brightness(img)
		return enhancer.enhance(factor)

	def __adjust_contrast(self, img: Image.Image) -> Image.Image:
		factor = random.uniform(0.7, 1.3)
		enhancer = ImageEnhance.Contrast(img)
		return enhancer.enhance(factor)

	def __shift_color(self, img: Image.Image) -> Image.Image:
		arr = np.array(img).astype(np.int16)
		shift = random.randint(-100, 100)
		i = random.randint(0, 2)

		arr[..., i] = np.clip(arr[..., i] + shift, 0, 255)
		return Image.fromarray(arr.astype(np.uint8))

	def __invert_color(self, img: Image.Image) -> Image.Image:
		arr = np.array(img)
		arr[..., :3] = 255 - arr[..., :3]
		return Image.fromarray(arr)

__all__ = ('ECGLeadAugmentor',)