import numpy as np
import matplotlib.pyplot as plt
from config import RADIUS, NUM_LANDMARKS


# ============================================================
#   TRAJECTORY PLOT (with desired path)
# ============================================================

def plot_paths(true_path, est_path, filename="robot_path.png"):
    tp = np.asarray(true_path)
    ep = np.asarray(est_path)

    plt.figure(figsize=(6, 6), dpi=150)


    phi = np.linspace(0, 2 * np.pi, 400)
    cx = RADIUS * np.cos(phi)
    cy = RADIUS * np.sin(phi)
    plt.plot(cx, cy, "g--", linewidth=2, alpha=0.7, label="Desired Path")


    plt.plot(tp[:, 0], tp[:, 1], "r-", linewidth=2.2, label="True Path")

    plt.plot(ep[:, 0], ep[:, 1], "k--", linewidth=2.0, label="EKF Estimated Path")

    margin = 2.0
    plt.xlim(-RADIUS - margin, RADIUS + margin)
    plt.ylim(-RADIUS - margin, RADIUS + margin)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Robot Trajectory: True vs EKF vs Desired Path")
    plt.axis("equal")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()



# ============================================================
#   MSE PLOT
# ============================================================

def plot_mse(mse_list, filename="mse_over_time.png"):
    mse = np.asarray(mse_list)

    plt.figure(figsize=(6, 4), dpi=150)
    plt.plot(mse, linewidth=2)
    plt.xlabel("Time Step")
    plt.ylabel("Position MSE [m²]")
    plt.title("Mean Square Error of Robot Position")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()



# ============================================================
#   LANDMARK ERROR PLOT
# ============================================================

def plot_landmark_errors(mu, lm_initialized, true_landmarks, filename="landmark_errors.png"):
    errors = []
    for j in range(NUM_LANDMARKS):
        if lm_initialized[j]:
            lm_true = true_landmarks[j]
            lm_est  = mu[3 + 2*j : 3 + 2*j + 2]
            errors.append(np.linalg.norm(lm_true - lm_est))
        else:
            errors.append(np.nan)  # not initialized

    plt.figure(figsize=(6, 4), dpi=150)
    plt.bar(np.arange(NUM_LANDMARKS), errors, color="#007BFF")
    plt.xlabel("Landmark Index")
    plt.ylabel("Error [m]")
    plt.title("Final Landmark Estimation Error")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()



# ============================================================
#   HEADING PLOT (TRUE vs EKF)
# ============================================================

def plot_heading(true_heading, est_heading, filename="heading_plot.png"):
    plt.figure(figsize=(6, 4), dpi=150)
    plt.plot(true_heading, "r-", linewidth=2, label="True Heading")
    plt.plot(est_heading, "k--", linewidth=1.8, label="EKF Heading")
    plt.xlabel("Time Step")
    plt.ylabel("Heading [rad]")
    plt.title("True vs Estimated Heading")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()



# ============================================================
#   SCATTER PLOT OF LANDMARKS
# ============================================================

def plot_landmark_scatter(true_landmarks, est_landmarks, filename="landmark_scatter.png"):
    plt.figure(figsize=(6, 6), dpi=150)
    plt.scatter(true_landmarks[:,0], true_landmarks[:,1],
                s=120, c="red", edgecolors="black", label="True LM")
    plt.scatter(est_landmarks[:,0], est_landmarks[:,1],
                s=120, c="#007BFF", edgecolors="black", label="Estimated LM")
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("True vs Estimated Landmarks")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
