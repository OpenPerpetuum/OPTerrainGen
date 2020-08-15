import numpy as np


class Filter:
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def do_filter(self, array):
        raise NotImplementedError()

    def do_filter_at(self, x, y):
        raise NotImplementedError()


class CachedFilter(Filter):
    _EMPTY_CACHE_ENTRY = -1

    def __init__(self, w, h):
        self._cache = np.ndarray(shape=(w, h), dtype=np.float)
        self._cache.fill(self._EMPTY_CACHE_ENTRY)
        super(CachedFilter, self).__init__(w, h)

    def generate_value_at(self, x, y):
        raise NotImplementedError()

    def do_filter_at(self, x, y):
        if 0 <= x < self.w and 0 <= y < self.h:
            x = int(x)
            y = int(y)
            if self._cache[x, y] == self._EMPTY_CACHE_ENTRY:
                self._cache[x, y] = self.generate_value_at(x, y)
            return self._cache[x, y]
        return self.generate_value_at(x, y)
