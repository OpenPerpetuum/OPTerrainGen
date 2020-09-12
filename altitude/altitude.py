import struct
from utils import utils
import os


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


def save_all_layers(array, width, height, island_num):
    island_str = str(island_num).zfill(4)
    if not os.path.exists(island_str):
        os.makedirs(island_str)
    save_empty_layer(width, height, bytes.fromhex('00 00 00 83 00 00 08'), island_str, 'plants', island_str)
    save_empty_layer(width, height, bytes.fromhex('00 00'), island_str, 'control', island_str)
    save_empty_layer(width, height, bytes.fromhex('00 00'), island_str, 'blocks', island_str)
    save_altitude(array, width, height, island_str, island_str)


def save_altitude(array, width, height, island_label, folder):
    file = os.path.join(folder, 'altitude.' + island_label + '.bin')
    with open(file, 'wb') as f:
        for x in range(width):
            for y in range(height):
                alt = int(array[x, y]) << 5
                slope = compute_slope(x, y, width, height, array)
                data = alt | slope
                byte_data = data.to_bytes(2, byteorder='little', signed=False)
                f.write(struct.pack('2s', byte_data))


def save_empty_layer(width, height, byte_pattern, island_label, layer_name, folder):
    file = os.path.join(folder, layer_name + '.' + island_label + '.bin')
    bytes_len = len(byte_pattern)
    pack_pattern = str(bytes_len) + 's'
    with open(file, 'wb') as f:
        for x in range(width):
            for y in range(height):
                f.write(struct.pack(pack_pattern, byte_pattern))
