from filters.filter import CachedFilter
from opensimplex import OpenSimplex


class Noise(CachedFilter):

    def __init__(self, w, h, SEED=3):
        self.plex = OpenSimplex(seed=SEED)
        self.scales = (200., 125., 75., 45., 35., 25., 15., 7., 3., 1.)
        self.weights = (200., 150., 100., 50., 25., 15., 7., 3., 2., 1.)
        temp_scales = []
        temp_weights = []
        for i in range(len(self.scales)):
            if self.scales[i] > w * 0.5:
                continue
            temp_scales.append(self.scales[i])
            temp_weights.append(self.weights[i])
        self.scales = tuple(temp_scales)
        self.weights = tuple(temp_weights)
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
        # return ((sum(z_factors) / self.sum_weights) + 1.0) / 2.0  # normalize to 0-1
