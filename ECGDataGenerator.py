import numpy as np
from PIL import Image
import random

from ECGLeadAugmentor import ECGLeadAugmentor
from ECGLeadGenerator import ECGLeadGenerator
from ECGSheetAugmentor import ECGSheetAugmentor
from ECGSheetLinker import ECGSheetLinker

class ECGDataGenerator:
	__MM_PER_SEC = (25, 50)
	__LABELS_COUNT_RANGE = (3, 15)
	__LABELS_LENGTH_RANGE = (5, 30)
	__LABELS_LETTERS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,:;!?-_\'"/\\()[]{}'
	__LABELS_MM_PER_SEC_COUNT_RANGE = (2, 5)
	__LABELS_MM_PER_SEC = ('{} mms', '{} mmps', '{} mm/s', '{} mm/sec', '{} mm per sec', '{} mm per second')

	__SECONDS = 8
	__SAMPLING_RATE = 100

	def __init__(self) -> None:
		self.__lead_generator = ECGLeadGenerator()
		self.__lead_augmentor = ECGLeadAugmentor()
		self.__sheet_linker = ECGSheetLinker()
		self.__sheet_augmentor = ECGSheetAugmentor()

	def generate(self) -> tuple[Image.Image, dict[str, dict[str, np.ndarray | tuple[int, int, int, int]]]]:
		mm_per_sec = random.choice(self.__MM_PER_SEC)
		self.__lead_generator.mm_per_sec = mm_per_sec

		labels = [
			''.join(random.choices(self.__LABELS_LETTERS, k=random.randint(*self.__LABELS_LENGTH_RANGE)))
			for _ in range(random.randint(*self.__LABELS_COUNT_RANGE))
		]

		for _ in range(random.randint(*self.__LABELS_MM_PER_SEC_COUNT_RANGE)):
			mmps_label = random.choice(self.__LABELS_MM_PER_SEC).format(mm_per_sec)
			mmps_variant = random.choice([mmps_label.lower(), mmps_label.upper(), mmps_label.capitalize()])
			labels.append(mmps_variant)

		leads = self.__lead_generator.generate_random_leads()
		augmented = {lead_name: self.__lead_augmentor.augment(lead) for lead_name, (lead, _) in leads.items()}
		composed, layout = self.__sheet_linker.compose(augmented, labels)
		image, new_layout = self.__sheet_augmentor.augment(composed, layout)
		gray = image.convert('L')

		result = {
			lead_name: {
				'signal': np.zeros(self.__SECONDS * self.__SAMPLING_RATE),
				'bbox': (-1, -1, -1, -1),
				'percentage': 0,
			} for lead_name in self.__lead_generator.LEAD_NAMES
		}

		for key, (x_start, _, x_end, _) in layout.items():
			if key.startswith('label') or key not in new_layout:
				continue
			elif key.startswith('rhythm'):
				lead_name = key[7:].split('_')[0]
			else:
				lead_name = key

			width = x_end - x_start
			percentage = width / self.__lead_generator.IMG_SIZE[0]
			if percentage > result[lead_name]['percentage']:
				duration = width / self.__lead_generator.pixels_per_mm / mm_per_sec
				values = int(duration * self.__SAMPLING_RATE)

				result[lead_name]['signal'][:values] = leads[lead_name][1][:values]
				result[lead_name]['bbox'] = new_layout[key]
				result[lead_name]['percentage'] = percentage

		for data in result.values():
			del data['percentage']

		return (gray, result)

__all__ = ('ECGDataGenerator',)