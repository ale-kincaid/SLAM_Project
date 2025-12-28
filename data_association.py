import numpy as np
from measurement import measurement_model
from utils import wrap_angle
from config import Q_measure, NUM_LANDMARKS, MAHA_THRESHOLD


def nearest_neighbor(meas, mu, lm_initialized):
    robot = mu[0:3]
    R_inv = np.linalg.inv(Q_measure)

    best_idx = None
    best_d2 = np.inf

    for i in range(NUM_LANDMARKS):
        if not lm_initialized[i]:
            continue

        lm = mu[3 + 2 * i : 3 + 2 * i + 2]
        z_hat = measurement_model(robot, lm)

        innov = meas - z_hat
        innov[1] = wrap_angle(innov[1])

        d2 = innov.T @ R_inv @ innov
        if d2 < best_d2:
            best_d2 = d2
            best_idx = i

    if best_idx is not None and best_d2 < MAHA_THRESHOLD:
        return best_idx
    else:
        return None
