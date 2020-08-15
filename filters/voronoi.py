from filters.filter import Filter
from utils import utils
import numpy as np


class Voronoi(Filter):

    def __init__(self, w, h, num_cells, margin=10):
        super(Voronoi, self).__init__(w, h)
        self.height_func = lambda x: x
        self.cells = self._generate_cells(num_cells, margin)
        self._min_dist = self.w * self.h

    def _generate_cells(self, num_cells, margin):
        poi_pts = np.random.poisson(num_cells)
        x = margin + np.random.uniform(0, 1, poi_pts) * self.w - (2 * margin)
        y = margin + np.random.uniform(0, 1, poi_pts) * self.h - (2 * margin)
        return np.stack((x, y), axis=1)

    def set_height_func(self, f):
        self.height_func = f

    def do_filter(self, array):
        voronoi_z = np.ndarray(shape=(self.w, self.h), dtype=np.float)
        for x in range(self.w):
            for y in range(self.h):
                voronoi_z[x, y] = self.do_filter_at(x, y)
        return voronoi_z

    def do_filter_at(self, x, y):
        min_dist = self._min_dist
        closest_cell = None
        for cell in self.cells:
            d = utils.dist((x, y), cell)
            if d <= min_dist:
                min_dist = d
                closest_cell = cell
        if closest_cell is None:
            closest_cell = (x, y)
        return self.height_func(closest_cell)
