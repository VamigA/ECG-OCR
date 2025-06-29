from multiprocessing import cpu_count, Pool
import os
from tqdm import tqdm

from generator.ECGLeadAugmentor import ECGLeadAugmentor
from generator.ECGLeadGenerator import ECGLeadGenerator
from generator.ECGSheetAugmentor import ECGSheetAugmentor
from generator.ECGSheetLinker import ECGSheetLinker

OUTPUT_DIR = 'generated_leads'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def init_generators():
	global lead_generator, lead_augmentor, sheet_linker, sheet_augmentor

	lead_generator = ECGLeadGenerator()
	lead_augmentor = ECGLeadAugmentor()
	sheet_linker = ECGSheetLinker()
	sheet_augmentor = ECGSheetAugmentor()

def generate_one(i):
	leads = lead_generator.generate_random_leads()

	composed, _ = sheet_linker.compose(
		{lead_name: lead_augmentor.augment(lead) for lead_name, (lead, _) in leads.items()},
		['Normal Sinus Rhythm', 'Atrial Fibrillation', 'Ventricular Tachycardia'],
	)

	image, _ = sheet_augmentor.augment(composed, {})
	image.save(os.path.join(OUTPUT_DIR, f'sheet_{i:04}.png'))

if __name__ == '__main__':
	num_images = 200
	num_workers = cpu_count()

	with Pool(num_workers, init_generators) as pool:
		tuple(tqdm(pool.imap_unordered(generate_one, range(num_images)), total=num_images))