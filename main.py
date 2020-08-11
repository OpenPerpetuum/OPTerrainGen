from opensimplex import OpenSimplex
from PIL import Image
import numpy as np
import math

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
    arr = np.ndarray(shape=(w, h))
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
            arr[x, y] = bound(z) * 255.0

    im = Image.fromarray(arr)
    im.show()


if __name__ == '__main__':
    go()
