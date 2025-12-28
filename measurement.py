import numpy as np
from utils import wrap_angle

def measurement_model(state, landmark):
    x, y, th = state
    lx, ly = landmark

    dx = lx - x
    dy = ly - y
    r = np.hypot(dx, dy)
    b = wrap_angle(np.arctan2(dy, dx) - th)

    return np.array([r, b])


def jacobian_measure(state, landmark):
    x, y, th = state
    lx, ly = landmark

    dx = lx - x
    dy = ly - y
    q = dx**2 + dy**2
    r = np.sqrt(q)

    if r < 1e-6:
        r = 1e-6
        q = r**2

    H_r = np.array([
        [-dx / r, -dy / r, 0.0],
        [ dy / q, -dx / q, -1.0],
    ])

    H_l = np.array([
        [ dx / r,  dy / r],
        [-dy / q,  dx / q],
    ])

    return H_r, H_l
