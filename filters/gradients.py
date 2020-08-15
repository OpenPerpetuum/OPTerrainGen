import numpy as np
import math
from utils import utils
from filters.filter import CachedFilter


class Gradient(CachedFilter):

    def __init__(self, w, h):
        super(Gradient, self).__init__(w, h)
        self.gradient = self._generate_gradient(w, h)

    def _generate_gradient(self, w, h):
        arr = np.ndarray(shape=(w, h), dtype=np.float)
        for x in range(w):
            for y in range(h):
                arr[x, y] = utils.bound(self.do_filter_at(x, y))
        return utils.scale_to_bounds(arr)

    def generate_value_at(self, x, y):
        sq_grad = utils.square_gradient(x, y, self.w, self.h)
        cir_grad = utils.circular_gradient(x, y, self.w, self.h)
        grad = sq_grad * cir_grad
        sin = math.sin(-utils.half_pi + grad * utils.half_pi) / 2.0 + 0.5
        power = pow(grad, 0.5)
        return (power + sin) / 2.0
