import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from config import (
    DT,
    STEPS,
    NUM_LANDMARKS,
    V_CMD,
    OMEGA_CMD,
    FOV,
    MAX_RANGE,
    Q_measure,
    LANDMARK_POSITIONS,
    INIT_LM_COV,
)
from video_export import save_video
from utils import wrap_angle
from motion import motion_model
from measurement import measurement_model
from ekf import ekf_predict, ekf_update
from data_association import nearest_neighbor
from visualization import make_figure, draw_frame
from plots import (
    plot_paths,
    plot_mse,
    plot_landmark_scatter,
    plot_landmark_errors,
    plot_heading,
)


def main():
    np.random.seed(0)


    landmarks_true = LANDMARK_POSITIONS.copy()

    from config import RADIUS
    true_state = np.array([RADIUS, 0.0, np.pi / 2])


    n_state = 3 + 2 * NUM_LANDMARKS
    mu = np.zeros(n_state)


    mu[0:3] = true_state + np.array([0.25, -0.15, 0.12])


    Sigma = np.eye(n_state) * 1e4
    Sigma[0:3, 0:3] = np.eye(3) * 0.4

    lm_initialized = [False] * NUM_LANDMARKS


    true_path = []
    est_path = []
    mse_list = []

    true_heading_list = []
    est_heading_list = []

    fig, ax = make_figure()
    frames = []

    meas_std = np.sqrt(np.diag(Q_measure))

    for k in range(STEPS):

        # v_noisy = V_CMD + np.random.randn() * 0.10
        # w_noisy = OMEGA_CMD + np.random.randn() * np.deg2rad(3.5)
        #
        # u_true = np.array([v_noisy, w_noisy])
        # true_state = motion_model(true_state, u_true)
        #
        # true_path.append(true_state[:2].copy())


        v_noisy = V_CMD + np.random.randn() * 0.04
        w_noisy = OMEGA_CMD + np.random.randn() * np.deg2rad(2.0)

        lateral_noise = 0.006 * np.sin(0.04 * k)
        true_state[0] += lateral_noise

        u_true = np.array([v_noisy, w_noisy])
        true_state = motion_model(true_state, u_true)
        true_path.append(true_state[:2].copy())


        u_cmd = np.array([V_CMD, OMEGA_CMD])
        mu, Sigma = ekf_predict(mu, Sigma, u_cmd)
        est_path.append(mu[0:2].copy())

        measurements = []

        for i, lm in enumerate(landmarks_true):
            z_true = measurement_model(true_state, lm)  # [r, b]
            r, b = z_true

            if r > MAX_RANGE or abs(b) > FOV / 2:
                continue

            z_noisy = np.array([
                r + np.random.randn() * meas_std[0],
                b + np.random.randn() * meas_std[1],
            ])
            z_noisy[1] = wrap_angle(z_noisy[1])
            measurements.append((i, z_noisy))

        for _, z in measurements:
            assoc_idx = nearest_neighbor(z, mu, lm_initialized)

            if assoc_idx is None:
                try:
                    j = lm_initialized.index(False)
                except ValueError:
                    continue

                r, b = z
                x, y, th = mu[0:3]

                lx = x + r * np.cos(th + b)
                ly = y + r * np.sin(th + b)

                lm_idx = 3 + 2 * j
                mu[lm_idx:lm_idx + 2] = np.array([lx, ly])

                Sigma[lm_idx:lm_idx + 2, lm_idx:lm_idx + 2] = np.eye(2) * INIT_LM_COV

                lm_initialized[j] = True
                continue

            mu, Sigma = ekf_update(mu, Sigma, z, assoc_idx)

        pos_err = true_state[:2] - mu[0:2]
        mse_list.append(pos_err @ pos_err)

        true_heading_list.append(true_state[2])
        est_heading_list.append(mu[2])

        draw_frame(
            ax,
            true_state,
            mu,
            landmarks_true,
            lm_initialized,
            true_path,
            est_path,
        )
        fig.canvas.draw()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))

    frames[0].save(
        "ekf_slam_simulation.gif",
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 * DT),
        loop=0,
    )


    plot_paths(true_path, est_path, "robot_path.png")
    plot_mse(mse_list, "mse_over_time.png")


    est_landmarks = []
    for j in range(NUM_LANDMARKS):
        if lm_initialized[j]:
            idx = 3 + 2 * j
            est_landmarks.append(mu[idx:idx + 2])
        else:
            est_landmarks.append([np.nan, np.nan])
    est_landmarks = np.array(est_landmarks)

    plot_landmark_scatter(
        landmarks_true,
        est_landmarks,
        filename="landmark_scatter.png",
    )

    plot_landmark_errors(
        mu,
        lm_initialized,
        landmarks_true,
        filename="landmark_errors.png",
    )

    plot_heading(
        true_heading_list,
        est_heading_list,
        filename="heading_plot.png",
    )

    fps = int(1.0 / DT)
    save_video(frames, filename="slam_video.mp4", fps=fps)

    print(
        "Saved: ekf_slam_simulation.gif, robot_path.png, mse_over_time.png, "
        "landmark_scatter.png, landmark_errors.png, heading_plot.png"
    )


if __name__ == "__main__":
    main()
