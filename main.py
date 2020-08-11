from opensimplex import OpenSimplex
from PIL import Image
import numpy as np
import math
import struct

SEED = 3

half_pi = math.pi / 2.0

w = 512
h = 512

scales = (155, 50, 20, 10, 5, 2)
weights = (155, 50, 20, 10, 5, 2)

gradient_weight = 100

plex = OpenSimplex(seed=SEED)


def square_gradient(x, y, width, height):
    hw = width / 2.0
    hh = height / 2.0
    vx = (hw - abs(hw - x)) / hw
    vy = (hh - abs(hh - y)) / hh
    return min(vx, vy)


def bound(x, lower=0.0, upper=1.0):
    return min(max(x, lower), upper)


def go():
    arr = np.ndarray(shape=(w, h), dtype=np.int32)
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
            z = sum(z_factors) / (sum_weights + gradient_weight)
            z *= math.sin(sq_grad * half_pi)
            z *= pow(sq_grad, 2.0) + 0.5
            arr[x, y] = int(bound(z) * 255.0)
    im = Image.fromarray(arr)
    save_altitude(arr)
    im.show()


def convert_to_altitude(array):
    for x in range(w):
        for y in range(h):
            array[x, y] = int(array[x, y]) << 5
    return array


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
