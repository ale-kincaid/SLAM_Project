import numpy as np
from config import DT
from utils import wrap_angle

def motion_model(state, control):
    x, y, th = state
    v, w = control

    x_new = x + v * np.cos(th) * DT
    y_new = y + v * np.sin(th) * DT
    th_new = wrap_angle(th + w * DT)

    return np.array([x_new, y_new, th_new])


def jacobian_motion_state(state, control):
    _, _, th = state
    v, _ = control

    Fx = np.array([
        [1.0, 0.0, -v * np.sin(th) * DT],
        [0.0, 1.0,  v * np.cos(th) * DT],
        [0.0, 0.0, 1.0],
    ])
    return Fx
