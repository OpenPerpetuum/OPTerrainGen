import math

half_pi = math.pi / 2.0


def circular_gradient(x, y, width, height):
    hw = width / 2.0
    hh = height / 2.0
    d = dist((x, y), (hw, hh))
    diag = math.sqrt(math.pow(hw, 2) + math.pow(hh, 2))
    return bound(1.0 - normalize(d, 0, diag, 0.0, 1.0), 0.0, 1.0)


def square_gradient(x, y, width, height):
    hw = width / 2.0
    hh = height / 2.0
    vx = (hw - abs(hw - x)) / hw
    vy = (hh - abs(hh - y)) / hh
    return bound(min(vx, vy), 0.0, 1.0)


def dist(pt_a, pt_b):
    d = [math.pow(a - b, 2) for a, b in zip(pt_a, pt_b)]
    return math.sqrt(sum(d))


def normalize(value, min_val, max_val, lower, upper):
    start_domain = max_val - min_val
    target_domain = upper - lower
    scale = target_domain / start_domain
    value -= (min_val - lower)
    value *= scale
    return value


def bound(x, lower=0.0, upper=1.0):
    return min(max(x, lower), upper)


def scale_to_bounds(array, lower=0.0, upper=1.0):
    min_val = array.min()
    max_val = array.max()
    starting_domain = max_val - min_val
    target_domain = upper - lower
    scale = target_domain / starting_domain
    array -= (min_val - lower)
    array *= scale
    return array
