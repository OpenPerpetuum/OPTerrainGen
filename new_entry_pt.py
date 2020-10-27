from filters.gradients import Gradient
from filters.noise import Noise
from filters.smooth import Smooth
from filters.voronoi import Voronoi
import numpy as np
from PIL import Image
from utils import utils
from altitude import altitude


def test_new_pipeline(island_num):
    w = 256
    h = 256
    SEED = island_num
    ALT_SCALE = 200
    WATER_LEVEL = 0.1

    gradient_weight = 3.5
    voronoi_weight = 4.0
    noise_weight = 4.0
    weight_sum = sum((gradient_weight, voronoi_weight, noise_weight))

    grad = Gradient(w, h)
    noise = Noise(w, h, SEED)
    smooth = Smooth(w, h, 9)
    voro = Voronoi(w, h, 25, 15)

    def height_from_gradient(pt):
        return grad.do_filter_at(pt[0], pt[1])

    def height_from_noise(pt):
        return noise.do_filter_at(pt[0], pt[1])

    def height_from_average(pt):
        g = height_from_gradient(pt)
        n = height_from_noise(pt)
        return (g + n) / 2.0

    def height_from_weighted(pt, gw=2.0, nw=2.0):
        g = height_from_gradient(pt) * gw
        n = height_from_noise(pt) * nw
        return (g + n) / (gw + nw)

    voro.set_height_func(height_from_weighted)

    zs = np.ndarray(shape=(w, h), dtype=np.float)

    plataeus = voro.do_filter(zs)
    smoothed_plataeus = smooth.do_filter(plataeus)

    for x in range(w):
        for y in range(h):
            g = grad.do_filter_at(x, y) * gradient_weight
            n = noise.do_filter_at(x, y) * noise_weight
            v = smoothed_plataeus[x, y] * voronoi_weight
            z = ((n + v + g)/weight_sum) * g - WATER_LEVEL
            z = max(z, 0)
            zs[x, y] = z

    arr = utils.scale_to_bounds(zs, 0.0, 255.0).astype(np.int32)
    im = Image.fromarray(arr)
    arr = utils.scale_to_bounds(zs, 0.0, ALT_SCALE).astype(np.int32)
    altitude.save_all_layers(arr, w, h, island_num)
    im.show()


if __name__ == '__main__':
    test_new_pipeline(80)
