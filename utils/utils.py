import math


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
