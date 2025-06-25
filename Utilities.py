from PIL import ImageFont

class staticproperty(property):
	def __get__(self, *_):
		return self.fget()

class classproperty(property):
	def __get__(self, _, objtype):
		return self.fget(objtype)

_font_cache = {}

def get_font(path, size):
	key = (path, size)
	if key not in _font_cache:
		try:
			_font_cache[key] = ImageFont.truetype(path, size)
		except:
			_font_cache[key] = ImageFont.load_default()

	return _font_cache[key]

__all__ = ('staticproperty', 'classproperty', 'get_font')