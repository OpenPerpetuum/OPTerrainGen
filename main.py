from opensimplex import OpenSimplex
from PIL import Image
from utils import utils
from altitude import altitude
import numpy as np
import math

SEED = 3

ALT_SCALE = 400

half_pi = math.pi / 2.0

w = 512
h = 512

# scales = (35, 25, 15, 7, 3, 1)
# weights = (25, 15, 7, 3, 2, 1)

scales = (200, 125, 75, 45, 35, 25, 15, 7, 3, 1)
weights = (200, 150, 100, 50, 25, 15, 7, 3, 2, 1)

gradient_weight = 5
voronoi_weight = 4
noise_weight = 6

num_voronoi_pts = 100

plex = OpenSimplex(seed=SEED)


def _old():
    arr = np.ndarray(shape=(w, h), dtype=np.int32)
    region_z = voronoi_regions(w, h, num_voronoi_pts)
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
    zs = np.ndarray(shape=(w, h), dtype=np.float)
    grad = smooth(island_gradient(), 5)
    noi = noise()
    plataeus = voronoi_regions(w, h, num_voronoi_pts)
    smoothed_plataeus = smooth(plataeus, 4)
    for i in range(1):
        smoothed_plataeus = smooth(smoothed_plataeus, 3)
    for x in range(w):
        for y in range(h):
            g = grad[x, y]
            n = noi[x, y] * noise_weight
            v = smoothed_plataeus[x, y] * voronoi_weight
            z = (n + v) / (noise_weight + voronoi_weight)
            z *= g
            zs[x, y] = z

    arr = scale_to_bounds(zs, 0.0, 255.0).astype(np.int32)
    im = Image.fromarray(arr)
    arr = scale_to_bounds(zs, 0.0, ALT_SCALE).astype(np.int32)
    altitude.save_altitude(arr, w, h)
    im.show()


def noise():
    arr = np.ndarray(shape=(w, h), dtype=np.float)
    for x in range(w):
        for y in range(h):
            arr[x, y] = do_noise_at(x, y, scales, weights)
    return scale_to_bounds(arr)


def island_gradient():
    arr = np.ndarray(shape=(w, h), dtype=np.float)
    for x in range(w):
        for y in range(h):
            arr[x, y] = utils.bound(do_gradient_at(x, y))
    return scale_to_bounds(arr)


def scale_to_bounds(array, lower=0.0, upper=1.0):
    min_val = array.min()
    max_val = array.max()
    starting_domain = max_val - min_val
    target_domain = upper - lower
    scale = target_domain / starting_domain
    array -= (min_val - lower)
    array *= scale
    return array


def do_gradient_at(x, y):
    sq_grad = utils.square_gradient(x, y, w, h)
    cir_grad = utils.circular_gradient(x, y, w, h)
    grad = sq_grad * cir_grad
    sin = math.sin(-half_pi + grad * half_pi) / 2.0 + 0.5
    power = pow(grad, 0.5)
    return (power + sin) / 2.0


def do_noise_at(x, y, scale_list=scales, weight_list=weights):
    sum_weights = sum(weight_list)
    z_factors = []
    for i in range(len(scale_list)):
        scale = scale_list[i]
        weight = weight_list[i]
        noise_val = plex.noise2d(x / scale, y / scale)
        z_factors.append(noise_val * weight)
    z = sum(z_factors) / sum_weights
    return z


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
            continue
        for j in range(-radius, radius + 1):
            y_coord = y + j
            if y_coord < 0 or y_coord >= height:
                continue
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


def voronoi_regions(width, height, num_regions):
    voronoi_z = np.ndarray(shape=(width, height), dtype=np.float)
    cell_origins = generate_poisson_pts(num_regions)
    for x in range(w):
        for y in range(h):
            voronoi_z[x, y] = do_voronoi_at(x, y, cell_origins, height_from_average)
    return voronoi_z


def height_from_gradient(pt):
    return do_gradient_at(pt[0], pt[1])


def height_from_noise(pt):
    return do_noise_at(pt[0], pt[1])


def height_from_average(pt):
    g = do_gradient_at(pt[0], pt[1])
    n = do_noise_at(pt[0], pt[1])
    return (g + n) / 2.0


def do_voronoi_at(x, y, cells, func=height_from_gradient):
    min_dist = w * h
    closest_cell = None
    for cell in cells:
        d = utils.dist((x, y), cell)
        if d <= min_dist:
            min_dist = d
            closest_cell = cell
    if closest_cell is None:
        closest_cell = (x, y)
    return func(closest_cell)


if __name__ == '__main__':
    pipeline()
