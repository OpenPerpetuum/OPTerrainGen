from filters.gradients import Gradient
from filters.noise import Noise
from filters.smooth import Smooth
from filters.voronoi import Voronoi
import numpy as np
from PIL import Image
from utils import utils
from altitude import altitude


def test_new_pipeline():
    w = 512
    h = 512
    SEED = 3

    gradient_weight = 5
    voronoi_weight = 4
    noise_weight = 6


    grad = Gradient(w, h)
    noise = Noise(w, h)
    smooth = Smooth(w, h)
    voro = Voronoi(w, h, 25, 10)

    def height_from_gradient(pt):
        return grad.do_filter_at(pt[0], pt[1])

    def height_from_noise(pt):
        return noise.do_filter_at(pt[0], pt[1])

    def height_from_average(pt):
        g = height_from_gradient(pt)
        n = height_from_noise(pt)
        return (g + n) / 2.0

    voro.set_height_func(height_from_average)

    zs = np.ndarray(shape=(w, h), dtype=np.float)

    plataeus = voro.do_filter(zs)
    smoothed_plataeus = smooth.do_filter(plataeus)

    for x in range(w):
        for y in range(h):
            g = grad.do_filter_at(x, y)
            n = noise.do_filter_at(x, y) * noise_weight
            v = smoothed_plataeus[x, y] * voronoi_weight
            z = (n + v) / (noise_weight + voronoi_weight)
            z *= g
            zs[x, y] = z

    arr = utils.scale_to_bounds(zs, 0.0, 255.0).astype(np.int32)
    im = Image.fromarray(arr)
    # arr = utils.scale_to_bounds(zs, 0.0, ALT_SCALE).astype(np.int32)
    # altitude.save_altitude(arr, w, h)
    im.show()


if __name__ == '__main__':
    test_new_pipeline()
