from PIL import ImageFont

_font_cache = {}

def get_font(path, size):
	key = (path, size)
	if key not in _font_cache:
		try:
			_font_cache[key] = ImageFont.truetype(path, size)
		except:
			_font_cache[key] = ImageFont.load_default()

	return _font_cache[key]

__all__ = ('get_font',)