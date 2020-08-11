from opensimplex import OpenSimplex
from PIL import Image
from utils import utils
from altitude import altitude
import numpy as np
import math

SEED = 6

half_pi = math.pi / 2.0

w = 512
h = 512

scales = (200, 125, 75, 45, 35, 25, 15, 7)
weights = (200, 150, 100, 50, 25, 15, 7, 3)

gradient_weight = 3
voronoi_weight = 1
noise_weight = 3

num_voronoi_pts = utils.bound((w * h) / 10000, 1, 100)

plex = OpenSimplex(seed=SEED)


def go():
    arr = np.ndarray(shape=(w, h), dtype=np.int32)
    region_z = voronoi_regions(arr)
    for x in range(w):
        for y in range(h):
            z_factors = []
            sum_weights = sum(weights)
            for i in range(len(scales)):
                scale = scales[i]
                weight = weights[i]
                noise_val = plex.noise2d(x / scale, y / scale)
                z_factors.append(noise_val * weight)
            sq_grad = utils.square_gradient(x, y, w, h)
            sin = math.sin(sq_grad * half_pi)
            z_factors.append(sin * gradient_weight)
            z_factors.append(region_z[x, y] * voronoi_weight)
            z = sum(z_factors) / (sum_weights + gradient_weight + voronoi_weight)
            z *= math.sin(sq_grad * half_pi)
            z *= pow(sq_grad, 2.0) + 0.5
            arr[x, y] = int(utils.bound(z) * 1023.0)

    arr = smooth(arr, 10)
    im = Image.fromarray(arr)
    altitude.save_altitude(arr, w, h)
    im.show()


def pipeline():
    arr = np.ndarray(shape=(w, h), dtype=np.int32)
    plataeus = voronoi_regions(arr)
    smoothed_plataeus = smooth(plataeus, 6)
    noise_terrain = noise()
    island_terrain = island_gradient()
    for x in range(w):
        for y in range(h):
            z_factors = []
            sum_weights = gradient_weight + voronoi_weight + noise_weight
            z_factors.append(smoothed_plataeus[x, y] * voronoi_weight)
            z_factors.append(noise_terrain[x, y] * noise_weight)
            z_factors.append(island_terrain[x, y] * gradient_weight)
            z = sum(z_factors) / (sum_weights)
            arr[x, y] = int(utils.bound(z) * 512.0)
    im = Image.fromarray(arr)
    altitude.save_altitude(arr, w, h)
    im.show()


def noise():
    arr = np.ndarray(shape=(w, h), dtype=np.float)
    for x in range(w):
        for y in range(h):
            z_factors = []
            sum_weights = sum(weights)
            for i in range(len(scales)):
                scale = scales[i]
                weight = weights[i]
                noise_val = plex.noise2d(x / scale, y / scale)
                z_factors.append(noise_val * weight)
            z = sum(z_factors) / (sum_weights)
            arr[x, y] = utils.bound(z)
    return arr


def island_gradient():
    arr = np.ndarray(shape=(w, h), dtype=np.float)
    for x in range(w):
        for y in range(h):
            z_factors = []
            sq_grad = utils.square_gradient(x, y, w, h)
            sin = math.sin(sq_grad * half_pi)
            power = pow(sq_grad, 2.0) + 0.5
            z_factors.append(sin)
            z_factors.append(power)
            z = sum(z_factors) / len(z_factors)
            arr[x, y] = utils.bound(z)
    return arr


def smooth(array, radius=2):
    smoothed = np.ndarray(shape=array.shape, dtype=np.float)
    for x in range(w):
        for y in range(h):
            neighbors = manhattan_neighborhood(x, y, w, h, radius)
            smoothed[x, y] = average_list(get_zs(array, neighbors))
    return smoothed


def manhattan_neighborhood(x, y, width, height, radius=1):
    neighbors = set()
    for i in range(-radius, radius + 1):
        x_coord = x + i
        if x_coord < 0 or x_coord >= width:
            break
        for j in range(-radius, radius + 1):
            y_coord = y + j
            if y_coord < 0 or y_coord >= height:
                break
            neighbors.add((x_coord, y_coord))
    return neighbors


def get_zs(array, coords):
    zs = []
    for pt in coords:
        zs.append(array[pt])
    return zs


def average_list(values):
    if len(values) < 1:
        return 0
    return sum(values) / len(values)


def generate_poisson_pts(num_pts=10, margin=(w / 10)):
    poi_pts = np.random.poisson(num_pts)
    x = margin + np.random.uniform(0, 1, poi_pts) * w - (2 * margin)
    y = margin + np.random.uniform(0, 1, poi_pts) * h - (2 * margin)
    return np.stack((x, y), axis=1)


def voronoi_regions(array):
    voronoi_z = np.ndarray(shape=array.shape, dtype=np.float)
    cell_origins = generate_poisson_pts(num_voronoi_pts)
    for x in range(w):
        for y in range(h):
            min_dist = w * h
            closest_cell = None
            for cell_pt in cell_origins:
                d = utils.dist((x, y), cell_pt)
                if d <= min_dist:
                    min_dist = d
                    closest_cell = cell_pt
            voronoi_z[x, y] = utils.square_gradient(closest_cell[0], closest_cell[1], w, h)
    return voronoi_z


if __name__ == '__main__':
    pipeline()
