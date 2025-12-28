import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from config import RADIUS, NUM_LANDMARKS


def make_figure():
    fig, ax = plt.subplots(figsize=(6, 6))
    _style_axes(ax)
    return fig, ax


def _style_axes(ax):
    ax.clear()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linestyle=":", alpha=0.5)



def _draw_road(ax):

    inner_r = RADIUS - 0.7
    outer_r = RADIUS + 0.7
    theta = np.linspace(0, 2 * np.pi, 400)

    x_outer = outer_r * np.cos(theta)
    y_outer = outer_r * np.sin(theta)
    x_inner = inner_r * np.cos(theta[::-1])
    y_inner = inner_r * np.sin(theta[::-1])

    xs = np.concatenate([x_outer, x_inner])
    ys = np.concatenate([y_outer, y_inner])

    road_poly = Polygon(
        np.vstack([xs, ys]).T,
        closed=True,
        facecolor="lightgray",
        edgecolor="dimgray",
        linewidth=1.0,
        alpha=0.9,
        zorder=0,
    )
    ax.add_patch(road_poly)


def _draw_car(ax, pose, color, label=None, zorder=5):

    x, y, th = pose

    length = 0.9
    width  = 0.5

    body_pts = np.array([
        [ length/2,  width/2],
        [ length/2, -width/2],
        [-length/2, -width/2],
        [-length/2,  width/2],
    ])

    bumper_pts = np.array([
        [ length/2 + 0.15,  width/2],
        [ length/2 + 0.15, -width/2],
        [ length/2,        -width/2],
        [ length/2,         width/2],
    ])

    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s],
                  [s,  c]])

    body_world   = (R @ body_pts.T).T   + np.array([x, y])
    bumper_world = (R @ bumper_pts.T).T + np.array([x, y])

    body_poly = Polygon(
        body_world,
        closed=True,
        facecolor=color,
        edgecolor="k",
        linewidth=1.0,
        zorder=zorder,
        label=label,
    )
    bumper_poly = Polygon(
        bumper_world,
        closed=True,
        facecolor="darkgray",
        edgecolor="k",
        linewidth=0.8,
        zorder=zorder+0.1,
    )

    ax.add_patch(body_poly)
    ax.add_patch(bumper_poly)



def draw_frame(ax,
               true_state,
               mu,
               landmarks_true,
               lm_initialized,
               true_path,
               est_path):

    from config import NUM_LANDMARKS, RADIUS

    _style_axes(ax)

    _draw_road(ax)

    phi = np.linspace(0, 2 * np.pi, 300)
    circle_x = RADIUS * np.cos(phi)
    circle_y = RADIUS * np.sin(phi)
    ax.plot(circle_x, circle_y, "g--", linewidth=2, alpha=0.7,
            label="Desired circle")

    if len(true_path) > 1:
        tp = np.array(true_path)
        ax.plot(tp[:, 0], tp[:, 1],
                color="red", linewidth=2.5,
                label="True path")

    if len(est_path) > 1:
        ep = np.array(est_path)
        ax.plot(ep[:, 0], ep[:, 1],
                "k--", linewidth=2.0,
                label="EKF path")

    ax.scatter(
        landmarks_true[:, 0],
        landmarks_true[:, 1],
        c="red",
        s=90,
        marker="^",
        edgecolors="k",
        linewidths=0.5,
        label="True Landmark",
        zorder=3,
    )

    est_lms = []
    for j in range(NUM_LANDMARKS):
        if lm_initialized[j]:
            idx = 3 + 2 * j
            est_lms.append(mu[idx:idx + 2])
    if est_lms:
        est_lms = np.array(est_lms)
        ax.scatter(
            est_lms[:, 0],
            est_lms[:, 1],
            c="blue",
            s=80,
            marker="^",
            edgecolors="k",
            linewidths=0.5,
            label="Est Landmark",
            zorder=3.5,
        )


    _draw_car(ax, true_state, color="red",   label="True robot", zorder=6)
    _draw_car(ax, mu[0:3],    color="black", label="EKF robot",  zorder=7)


    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(),
              loc="upper right", fontsize=9, framealpha=0.9)
