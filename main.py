from opensimplex import OpenSimplex
from PIL import Image
import numpy as np
import math
import struct

SEED = 6

half_pi = math.pi / 2.0

w = 512
h = 512

scales = (200, 125, 75, 45, 35, 25, 15, 7)
weights = (200, 150, 100, 50, 25, 15, 7, 3)

gradient_weight = 200

voronoi_weight = 200

smooth_weight = 100

plex = OpenSimplex(seed=SEED)


def square_gradient(x, y, width, height):
    hw = width / 2.0
    hh = height / 2.0
    vx = (hw - abs(hw - x)) / hw
    vy = (hh - abs(hh - y)) / hh
    return min(vx, vy)


def dist(pt_a, pt_b, n_dimensions=2):
    sum_sqrs = 0
    for i in range(n_dimensions):
        sum_sqrs += math.pow(pt_a[i] - pt_b[i], 2)
    return math.sqrt(sum_sqrs)


def bound(x, lower=0.0, upper=1.0):
    return min(max(x, lower), upper)


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
            sq_grad = square_gradient(x, y, w, h)
            sin = math.sin(sq_grad * half_pi)
            z_factors.append(sin * gradient_weight)
            z_factors.append(region_z[x, y] * voronoi_weight)
            z = sum(z_factors) / (sum_weights + gradient_weight + voronoi_weight)
            z *= math.sin(sq_grad * half_pi)
            z *= pow(sq_grad, 2.0) + 0.5
            arr[x, y] = int(bound(z) * 1023.0)

    arr = smooth(arr)
    im = Image.fromarray(arr)
    save_altitude(arr)
    im.show()


def smooth(array, radius=2):
    smoothed = np.ndarray(shape=array.shape, dtype=array.dtype)
    for x in range(w):
        for y in range(h):
            neighbors = manhattan_neighborhood(x, y, w, h, radius)
            smoothed[x, y] = int(average_list(get_zs(array, neighbors)))
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
    cell_origins = generate_poisson_pts(25)
    for x in range(w):
        for y in range(h):
            min_dist = w * h
            closest_cell = None
            for cell_pt in cell_origins:
                d = dist((x, y), cell_pt)
                if d <= min_dist:
                    min_dist = d
                    closest_cell = cell_pt
            voronoi_z[x, y] = square_gradient(closest_cell[0], closest_cell[1], w, h)
    return voronoi_z


def compute_slope(x, y, width, height, array):
    x0 = bound((x + 1), 0, width - 1)
    y0 = bound((y + 1), 0, height - 1)
    a, b, c, d = array[x, y], array[x0, y], array[x0, y0], array[x, y0]
    e = (a + b) >> 1
    f = (b + c) >> 1
    g = (c + d) >> 1
    h = (d + e) >> 1
    i = (a + b + c + d) >> 2
    slope_byte = abs(i - a) + abs(i - b) + abs(i - c) + abs(i - d) + abs(i - e) + abs(i - f) + abs(i - g) + abs(i - h)
    slope_byte *= 2
    return int(bound(slope_byte, 0, 255))


def save_altitude(array):
    with open('altitude.bin', 'wb') as f:
        for x in range(w):
            for y in range(h):
                alt = int(array[x, y]) << 5
                slope = compute_slope(x, y, w, h, array)
                data = alt | slope
                f.write(struct.pack('<H', int(data)))


if __name__ == '__main__':
    go()
