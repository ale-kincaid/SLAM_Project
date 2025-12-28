import numpy as np
from motion import motion_model, jacobian_motion_state
from measurement import measurement_model, jacobian_measure
from utils import wrap_angle
from config import R_motion, Q_measure

def ekf_predict(mu, Sigma, u):
    n = len(mu)
    robot = mu[0:3]

    Fx = jacobian_motion_state(robot, u)

    F_big = np.eye(n)
    F_big[0:3, 0:3] = Fx

    R_big = np.zeros((n, n))
    R_big[0:3, 0:3] = R_motion

    mu[0:3] = motion_model(robot, u)
    Sigma[:] = F_big @ Sigma @ F_big.T + R_big

    return mu, Sigma


def ekf_update(mu, Sigma, meas, lm_idx, alpha_robot=1.0):
    n = len(mu)
    robot = mu[0:3]
    lm_state_idx = 3 + 2 * lm_idx
    lm = mu[lm_state_idx : lm_state_idx + 2]

    H_r, H_l = jacobian_measure(robot, lm)

    H = np.zeros((2, n))
    H[:, 0:3] = H_r
    H[:, lm_state_idx : lm_state_idx + 2] = H_l

    z_hat = measurement_model(robot, lm)
    innov = meas - z_hat
    innov[1] = wrap_angle(innov[1])

    S = H @ Sigma @ H.T + Q_measure
    K = Sigma @ H.T @ np.linalg.inv(S)   # (n x 2)

    K[0:3, :] *= alpha_robot

    mu[:] = mu + K @ innov
    mu[2] = wrap_angle(mu[2])

    I = np.eye(n)
    Sigma[:] = (I - K @ H) @ Sigma

    return mu, Sigma
