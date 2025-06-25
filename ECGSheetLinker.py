import itertools
import math
from PIL import Image, ImageDraw
import random

from Utilities import get_font

class ECGSheetLinker:
	__CANVAS_WIDTH_RANGE = (500, 1000)
	__CANVAS_HEIGHT_RANGE = (500, 1000)
	__BACKGROUND_COLORS = ((0, 0, 0, 0), (0, 0, 0, 255), (255, 200, 200, 255), (255, 255, 255, 255))
	__MARGIN_RANGE = (0, 20)
	__ROW_GAP_RANGE = (0, 20)
	__COLUMNS_RANGE = (1, 4)
	__COLUMN_GAP_RANGE = (0, 5)
	__RANDOM_PLACE_ATTEMPTS = 50
	__LABELS_FONTS = ('fonts/arial.ttf', 'fonts/times.ttf')
	__LABELS_SIZE_RANGE = (12, 24)
	__LABELS_COLORS = ((0, 0, 0), (50, 50, 50), (200, 0, 0))

	__canvas_size = None
	__background_color = None
	__standard_layout = None
	__margin_top = None
	__margin_right = None
	__margin_bottom = None
	__margin_left = None
	__row_gap = None
	__columns = None
	__column_gap = None
	__column_min_width = 200
	__rhythm_names = None
	__rhythm_position = None
	__shuffle_leads = True
	__lead_skip_probability = 0.05
	__layout_direction = None
	__labels_variants = True
	__labels_font = None
	__labels_size = None
	__labels_color = None

	def compose(self, leads: dict[str, Image.Image], labels: list[str]) -> tuple[Image.Image, dict[str, tuple[int, int, int, int]]]:
		background_color = random.choice(self.__BACKGROUND_COLORS) if self.__background_color is None else self.__background_color
		standard_layout = random.choice([False, True]) if self.__standard_layout is None else self.__standard_layout

		canvas_size = (
			random.randint(*self.__CANVAS_WIDTH_RANGE),
			random.randint(*self.__CANVAS_HEIGHT_RANGE) if not standard_layout else self.__CANVAS_HEIGHT_RANGE[1],
		) if self.__canvas_size is None else self.__canvas_size

		canvas = Image.new('RGBA', canvas_size, background_color)
		if standard_layout:
			layout = self.__compose_standard(canvas, leads)
			max_y = max(box[3] for box in layout.values())
			margin_bottom = random.randint(*self.__MARGIN_RANGE) if self.__margin_bottom is None else self.__margin_bottom
			canvas = canvas.crop((0, 0, canvas.width, max_y + margin_bottom))
		else:
			layout = self.__compose_random(canvas, leads)

		self.__draw_random_labels(canvas, labels, layout)
		return (canvas, layout)

	def __compose_standard(self, canvas: Image.Image, leads: dict[str, Image.Image]) -> dict[str, tuple[int, int, int, int]]:
		layout = {}
		margin_top = random.randint(*self.__MARGIN_RANGE) if self.__margin_top is None else self.__margin_top
		margin_left = random.randint(*self.__MARGIN_RANGE) if self.__margin_left is None else self.__margin_left
		margin_right = random.randint(*self.__MARGIN_RANGE) if self.__margin_right is None else self.__margin_right
		columns = random.randint(*self.__COLUMNS_RANGE) if self.__columns is None else self.__columns

		if self.__rhythm_names is None:
			rhythm_count = random.randint(0, 3)
			rhythm_names = random.choices(tuple(leads.keys()), k=rhythm_count)
		else:
			rhythm_count = len(self.__rhythm_names)
			rhythm_names = self.__rhythm_names

		if self.__rhythm_position is None:
			top_rhythm_count = random.randint(0, rhythm_count)
		else:
			top_rhythm_count = rhythm_count if self.__rhythm_position == 'top' else 0

		if self.__column_gap is None:
			column_gaps = tuple(random.randint(*self.__COLUMN_GAP_RANGE) for _ in range(columns - 1))
		else:
			column_gaps = (self.__column_gap,) * (columns - 1)

		container_width = canvas.width - margin_left - margin_right
		content_width = container_width - sum(column_gaps)
		available_width = content_width - self.__column_min_width * columns

		if available_width > 0:
			column_widths = []
			for _ in range(columns - 1):
				column_extra_width = random.randint(0, available_width)
				column_widths.append(self.__column_min_width + column_extra_width)
				available_width -= column_extra_width

			column_widths.append(self.__column_min_width + available_width)
		else:
			column_min_width = content_width // columns
			column_widths = (column_min_width,) * columns

		x_positions = []
		x_cursor = margin_left
		for width, gap in itertools.zip_longest(column_widths, column_gaps, fillvalue=0):
			x_positions.append(x_cursor)
			x_cursor += width + gap

		y_cursor = self.__place_rhythm_leads(canvas, {name: leads[name] for name in rhythm_names[:top_rhythm_count]},
			margin_left, margin_top, container_width, layout)

		column_y_cursors = [y_cursor] * columns
		lead_names = list(leads.keys())
		if self.__shuffle_leads:
			random.shuffle(lead_names)

		for i, name in enumerate(lead_names):
			if random.random() <= self.__lead_skip_probability:
				continue

			if self.__layout_direction == 'horizontal':
				column = i % columns
			elif self.__layout_direction == 'vertical':
				rows = math.ceil(len(lead_names) / columns)
				column = i // rows
			else:
				column = random.randint(0, columns - 1)

			x, y, width = x_positions[column], column_y_cursors[column], column_widths[column]
			_, box = self.__place_lead(canvas, leads[name], x, y, width, name, layout)
			row_gap = random.randint(*self.__ROW_GAP_RANGE) if self.__row_gap is None else self.__row_gap
			column_y_cursors[column] = box[3] + row_gap

		self.__place_rhythm_leads(canvas, {name: leads[name] for name in rhythm_names[top_rhythm_count:]},
			margin_left, max(column_y_cursors), container_width, layout)

		return layout

	def __place_rhythm_leads(self, canvas: Image.Image, rhythm_leads: dict[str, Image.Image], x: int, y: int, width: int,
			layout: dict[str, tuple[int, int, int, int]]) -> int:
		for name, lead in rhythm_leads.items():
			_, box = self.__place_lead(canvas, lead, x, y, width, f'rhythm_{name}', layout)
			row_gap = random.randint(*self.__ROW_GAP_RANGE) if self.__row_gap is None else self.__row_gap
			y = box[3] + row_gap

		return y

	def __compose_random(self, canvas: Image.Image, leads: dict[str, Image.Image]) -> dict[str, tuple[int, int, int, int]]:
		layout = {}
		if self.__rhythm_names is None:
			rhythm_count = random.randint(0, 3)
			rhythm_names = random.choices(tuple(leads.keys()), k=rhythm_count)
		else:
			rhythm_count = len(self.__rhythm_names)
			rhythm_names = self.__rhythm_names

		regular_leads = {name: lead for name, lead in leads.items() if random.random() > self.__lead_skip_probability}
		rhythm_leads = {f'rhythm_{name}': leads[name] for name in rhythm_names}

		for name, lead in itertools.chain(regular_leads.items(), rhythm_leads.items()):
			width = random.randint(self.__column_min_width, lead.width)
			for _ in range(self.__RANDOM_PLACE_ATTEMPTS):
				x = random.randint(0, canvas.width - width)
				y = random.randint(0, canvas.height - lead.height)
				if not self.__is_overlapping(layout, x, y, width, lead.height):
					self.__place_lead(canvas, lead, x, y, width, name, layout)
					break

		return layout

	def __place_lead(self, canvas: Image.Image, lead: Image.Image, x: int, y: int, width: int, name: str,
			layout: dict[str, tuple[int, int, int, int]]) -> tuple[str, tuple[int, int, int, int]]:
		if lead.width < width:
			x += random.randint(0, width - lead.width)
		elif lead.width > width:
			lead = lead.crop((0, 0, width, lead.height))

		if x >= canvas.width or y >= canvas.height:
			return ('', (x, y, x, y))

		canvas.paste(lead, (x, y), lead)

		for i in itertools.count():
			key = name if i == 0 else f'{name}_{i}'
			if key not in layout:
				x_end = min(x + lead.width, canvas.width)
				y_end = min(y + lead.height, canvas.height)
				layout[key] = (x, y, x_end, y_end)
				return (key, layout[key])

	def __draw_random_labels(self, canvas: Image.Image, labels: list[str], layout: dict[str, tuple[int, int, int, int]]) -> None:
		draw = ImageDraw.Draw(canvas)
		for label in labels:
			if self.__labels_variants:
				label = random.choice([label.lower(), label.upper(), label.capitalize()])

			file = random.choice(self.__LABELS_FONTS) if self.__labels_font is None else self.__labels_font
			size = random.randint(*self.__LABELS_SIZE_RANGE) if self.__labels_size is None else self.__labels_size
			color = random.choice(self.__LABELS_COLORS) if self.__labels_color is None else self.__labels_color
			font = get_font(file, size)

			for _ in range(self.__RANDOM_PLACE_ATTEMPTS):
				x = random.randint(0, canvas.width - size * 8)
				y = random.randint(0, canvas.height - size)
				if not self.__is_overlapping(layout, x, y, size * 8, size):
					draw.text((x, y), label, color, font)

					for i in itertools.count():
						key = f'label: {label}' if i == 0 else f'label_{i}: {label}'
						if key not in layout:
							x_end = min(x + size * 8, canvas.width)
							y_end = min(y + size, canvas.height)
							layout[key] = (x, y, x_end, y_end)
							break

					break

	def __is_overlapping(self, layout: dict[str, tuple[int, int, int, int]], x: int, y: int, w: int, h: int) -> bool:
		for x_start, y_start, x_end, y_end in layout.values():
			if x < x_end and x + w > x_start and y < y_end and y + h > y_start:
				return True

		return False

__all__ = ('ECGSheetLinker',)