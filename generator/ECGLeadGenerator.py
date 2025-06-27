import numpy as np
from PIL import Image, ImageDraw
import random
from scipy.interpolate import interp1d

from Utilities import staticproperty, classproperty, get_font

class ECGLeadGenerator:
	__IMG_WIDTH = 400
	__IMG_HEIGHT = 60
	__MM_PER_SEC = (25, 50)
	__GRID_WIDTHS = (1, 2)
	__GRID_COLORS = ((220, 220, 220), (255, 200, 200))
	__LINE_WIDTHS = (1, 2)
	__LINE_COLORS = ((0, 0, 0), (50, 50, 50), (200, 0, 0))
	__LEAD_NAME_X_RANGE = (0, 36)
	__LEAD_NAME_Y_RANGE = (0, 36)
	__LEAD_NAME_FONTS = ('generator/fonts/arial.ttf', 'generator/fonts/times.ttf')
	__LEAD_NAME_SIZE_RANGE = (12, 24)
	__LEAD_NAME_COLORS = ((0, 0, 0), (50, 50, 50), (200, 0, 0))

	__mm_per_sec = None
	__pixels_per_mm = 2
	__pixels_per_mv = 20
	__grid_width = None
	__grid_color = None
	__grid_step_large = 10
	__grid_step_small = 2
	__line_width = None
	__line_color = None
	__lead_name_variants = True
	__lead_name_x = None
	__lead_name_y = None
	__lead_name_font = None
	__lead_name_size = None
	__lead_name_color = None

	@classproperty
	def IMG_SIZE(cls):
		return (cls.__IMG_WIDTH, cls.__IMG_HEIGHT)

	@staticproperty
	def SIGNAL_TYPES():
		return ('sin', 'zigzag', 'triangle', 'noise')

	@staticproperty
	def LEAD_NAMES():
		return ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')

	@property
	def mm_per_sec(self):
		return self.__mm_per_sec

	@mm_per_sec.setter
	def mm_per_sec(self, mm_per_sec):
		if not isinstance(mm_per_sec, int) and mm_per_sec is not None:
			raise TypeError('The mm_per_sec argument must be of type int or none!')

		self.__mm_per_sec = mm_per_sec

	@property
	def pixels_per_mm(self):
		return self.__pixels_per_mm

	@property
	def pixels_per_mv(self):
		return self.__pixels_per_mv

	@staticmethod
	def generate_signal(signal_type='sin', scale_x=1, scale_y=1, duration=10, sampling_rate=100):
		t = np.linspace(0, duration, duration * sampling_rate) / scale_x

		if signal_type == 'sin':
			signal = np.sin(2 * np.pi * t)
		elif signal_type == 'zigzag':
			signal = 2 * (t % 1) - 1
		elif signal_type == 'triangle':
			signal = 2 * np.abs(2 * (t % 1) - 1) - 1
		elif signal_type == 'noise':
			coarse_rate = random.randint(5, 40)
			t_coarse = np.linspace(0, duration, duration * coarse_rate)
			noise = np.random.normal(0, 0.33, len(t_coarse))

			interpolator = interp1d(t_coarse, noise, 'cubic', fill_value='extrapolate')
			signal = interpolator(t)
		else:
			signal = np.zeros(len(t))

		return signal * scale_y

	def generate_lead_image(self, signal, lead_name='I', sampling_rate=100):
		img = Image.new('RGBA', (self.__IMG_WIDTH, self.__IMG_HEIGHT), 'white')
		draw = ImageDraw.Draw(img)

		self.__draw_grid(draw)
		self.__draw_line(draw, signal, sampling_rate)
		self.__draw_lead_name(draw, lead_name)

		return img

	def generate_random_lead(self, lead_name='I'):
		signal_type = random.choice(self.SIGNAL_TYPES)
		scale_x = random.gauss(1, 0.1)
		scale_y = random.gauss(1, 0.1)

		signal = self.generate_signal(signal_type, scale_x, scale_y)
		img = self.generate_lead_image(signal, lead_name)

		return (img, signal)

	def generate_random_leads(self):
		return {lead_name: self.generate_random_lead(lead_name) for lead_name in self.LEAD_NAMES}

	def __draw_grid(self, draw):
		width = random.choice(self.__GRID_WIDTHS) if self.__grid_width is None else self.__grid_width
		color_small = random.choice(self.__GRID_COLORS) if self.__grid_color is None else self.__grid_color
		color_large = tuple(value // 2 for value in color_small)

		for x in range(0, self.__IMG_WIDTH, self.__grid_step_small):
			if x % self.__grid_step_large != 0:
				draw.line((x, 0, x, self.__IMG_HEIGHT), color_small, width)
		for y in range(0, self.__IMG_HEIGHT, self.__grid_step_small):
			if y % self.__grid_step_large != 0:
				draw.line((0, y, self.__IMG_WIDTH, y), color_small, width)

		for x in range(0, self.__IMG_WIDTH, self.__grid_step_large):
			draw.line((x, 0, x, self.__IMG_HEIGHT), color_large, width)
		for y in range(0, self.__IMG_HEIGHT, self.__grid_step_large):
			draw.line((0, y, self.__IMG_WIDTH, y), color_large, width)

	def __draw_line(self, draw, signal, sampling_rate=250):
		mm_per_sec = random.choice(self.__MM_PER_SEC) if self.__mm_per_sec is None else self.__mm_per_sec
		width = random.choice(self.__LINE_WIDTHS) if self.__line_width is None else self.__line_width
		color = random.choice(self.__LINE_COLORS) if self.__line_color is None else self.__line_color

		duration = len(signal) / sampling_rate
		pixels = duration * mm_per_sec * self.__pixels_per_mm

		x = np.linspace(0, pixels, len(signal))
		y = self.__IMG_HEIGHT // 2 - self.__pixels_per_mv * signal
		xy = tuple(zip(x.astype(int), y.astype(int)))
		draw.line(xy, color, width)

	def __draw_lead_name(self, draw, lead_name='I'):
		if self.__lead_name_variants:
			lead_name = random.choice([lead_name.lower(), lead_name.upper(), lead_name.capitalize()])

		x = random.randint(*self.__LEAD_NAME_X_RANGE) if self.__lead_name_x is None else self.__lead_name_x
		y = random.randint(*self.__LEAD_NAME_Y_RANGE) if self.__lead_name_y is None else self.__lead_name_y

		file = random.choice(self.__LEAD_NAME_FONTS) if self.__lead_name_font is None else self.__lead_name_font
		size = random.randint(*self.__LEAD_NAME_SIZE_RANGE) if self.__lead_name_size is None else self.__lead_name_size
		color = random.choice(self.__LEAD_NAME_COLORS) if self.__lead_name_color is None else self.__lead_name_color
		font = get_font(file, size)

		draw.text((x, y), lead_name, color, font)

__all__ = ('ECGLeadGenerator',)