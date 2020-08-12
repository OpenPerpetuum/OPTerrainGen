import math


def circular_gradient(x, y, width, height):
    hw = width / 2.0
    hh = height / 2.0
    d = dist((x, y), (hw, hh))
    diag = math.sqrt(math.pow(hw, 2) + math.pow(hh, 2))
    return bound(1.0-normalize(d, 0, diag, 0.0, 1.0), 0.0, 1.0)


def square_gradient(x, y, width, height):
    hw = width / 2.0
    hh = height / 2.0
    vx = (hw - abs(hw - x)) / hw
    vy = (hh - abs(hh - y)) / hh
    return bound(min(vx, vy), 0.0, 1.0)


def dist(pt_a, pt_b, n_dimensions=2):
    sum_sqrs = 0
    for i in range(n_dimensions):
        sum_sqrs += math.pow(pt_a[i] - pt_b[i], 2)
    return math.sqrt(sum_sqrs)


def normalize(value, min_val, max_val, lower, upper):
    start_domain = max_val-min_val
    target_domain = upper-lower
    scale = target_domain/start_domain
    value -= (min_val - lower)
    value *= scale
    return value


def bound(x, lower=0.0, upper=1.0):
    return min(max(x, lower), upper)
