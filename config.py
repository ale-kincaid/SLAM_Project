import numpy as np

DT = 0.1
STEPS = 500

RADIUS = 4.0
V_CMD = 1.0
OMEGA_CMD = V_CMD / RADIUS

NUM_LANDMARKS = 6
WORLD_HALF_SIZE = 10.0

# ----- RANDOM / FIXED LANDMARKS -----
USE_RANDOM_LANDMARKS = False
RANDOM_LM_SEED = 0

if not USE_RANDOM_LANDMARKS:
    LANDMARK_POSITIONS = np.array([
        [-6.0, -6.0],
        [ 6.0, -5.0],
        [ 7.0,  3.0],
        [-7.0,  3.0],
        [-3.0,  7.0],
        [ 3.0,  8.0],
    ])[:NUM_LANDMARKS]
else:
    rng = np.random.default_rng(RANDOM_LM_SEED)
    LANDMARK_POSITIONS = rng.uniform(
        low=-WORLD_HALF_SIZE * 0.8,
        high= WORLD_HALF_SIZE * 0.8,
        size=(NUM_LANDMARKS, 2)
    )


PROC_STD_X  = 0.07
PROC_STD_Y  = 0.07
PROC_STD_TH = np.deg2rad(2.5)
R_motion = np.diag([PROC_STD_X**2, PROC_STD_Y**2, PROC_STD_TH**2])


MEAS_STD_R = 0.05
MEAS_STD_B = np.deg2rad(1.2)
Q_measure = np.diag([MEAS_STD_R**2, MEAS_STD_B**2])


FOV = np.deg2rad(120.0)
MAX_RANGE = 13.0

MAHA_THRESHOLD = 35.0

INIT_LM_COV = 1.5
