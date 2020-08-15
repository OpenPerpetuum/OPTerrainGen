from filters.filter import CachedFilter
from opensimplex import OpenSimplex

SEED = 3


class Noise(CachedFilter):

    def __init__(self, w, h):
        self.plex = OpenSimplex(seed=SEED)
        self.scales = (200, 125, 75, 45, 35, 25, 15, 7, 3, 1)
        self.weights = (200, 150, 100, 50, 25, 15, 7, 3, 2, 1)
        super(Noise, self).__init__(w, h)
        self.passes = range(len(self.scales))
        self.sum_weights = sum(self.weights)

    def generate_value_at(self, x, y):
        z_factors = []
        for i in self.passes:
            scale = self.scales[i]
            weight = self.weights[i]
            noise_val = self.plex.noise2d(x / scale, y / scale)
            z_factors.append(noise_val * weight)
        return sum(z_factors) / self.sum_weights
