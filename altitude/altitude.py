import struct
from utils import utils


def compute_slope(x, y, width, height, array):
    x0 = utils.bound((x + 1), 0, width - 1)
    y0 = utils.bound((y + 1), 0, height - 1)
    a, b, c, d = array[x, y], array[x0, y], array[x0, y0], array[x, y0]
    e = (a + b) >> 1
    f = (b + c) >> 1
    g = (c + d) >> 1
    h = (d + e) >> 1
    i = (a + b + c + d) >> 2
    slope_byte = abs(i - a) + abs(i - b) + abs(i - c) + abs(i - d) + abs(i - e) + abs(i - f) + abs(i - g) + abs(i - h)
    slope_byte *= 2
    return int(utils.bound(slope_byte, 0, 255))


def save_altitude(array, width, height):
    with open('altitude.bin', 'wb') as f:
        for x in range(width):
            for y in range(height):
                alt = int(array[x, y]) << 5
                slope = compute_slope(x, y, width, height, array)
                data = alt | slope
                f.write(struct.pack('<H', int(data)))
