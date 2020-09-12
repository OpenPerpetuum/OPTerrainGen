import scipy as sp
import scipy.ndimage

from filters.filter import Filter


class Smooth(Filter):

    def __init__(self, w, h, radius):
        self.sigma = [radius, radius]
        super(Smooth, self).__init__(w, h)

    def do_filter(self, array):
        return sp.ndimage.filters.gaussian_filter(array, self.sigma, mode='constant')
