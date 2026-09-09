import copy

import matplotlib.pyplot as plt
import numpy as np

from models import rimless_wheel as model

PERTURBATION = 1e-4
THETA_RANGE = np.linspace(-1.5, 1.5, 25)
THETA_DOT_RANGE = np.linspace(-6.0, 6.0, 25)


def with_incline(params, incline_deg):
    updated = copy.deepcopy(params)
    updated["incline_angle"] = np.deg2rad(incline_deg)
    return updated


def with_num_spokes(params, num_spokes):
    updated = copy.deepcopy(params)
    updated["num_spokes"] = num_spokes
    updated["half_spoke_angle"] = np.pi / num_spokes
    return updated


def sweep_fixed_point_and_floquet(base_params, param_values, make_params):
    """For each parameter value, find the return map's fixed point and
    estimate the Floquet multiplier as the map's local slope there."""
    fixed_points = np.full(len(param_values), np.nan)
    floquet_multipliers = np.full(len(param_values), np.nan)
    for i, value in enumerate(param_values):
        params = make_params(base_params, value)
        try:
            fixed_point = model.find_fixed_point(params)
        except RuntimeError:
            continue
        mapped_minus = model.poincare_map(fixed_point - PERTURBATION, params)
        mapped_plus = model.poincare_map(fixed_point + PERTURBATION, params)
        fixed_points[i] = fixed_point
        floquet_multipliers[i] = (mapped_plus - mapped_minus) / (2 * PERTURBATION)
    return fixed_points, floquet_multipliers


def plot_fixed_point_and_floquet(
    param_values, fixed_points, floquet_multipliers, xlabel, title, filename
):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    axes[0].plot(param_values, fixed_points, "o-")
    axes[0].set_ylabel(r"fixed point $\dot\theta^*$ (rad/s)")
    axes[1].plot(param_values, floquet_multipliers, "o-")
    axes[1].axhline(1.0, color="gray", linestyle="--")
    axes[1].set_ylabel("Floquet multiplier")
    axes[1].set_xlabel(xlabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename)


def plot_roa_panels(base_params, param_values, make_params, label_fn, title, filename):
    fig, axes = plt.subplots(
        1, len(param_values), figsize=(5 * len(param_values), 4), sharey=True
    )
    for ax, value in zip(axes, param_values):
        params = make_params(base_params, value)
        rolling_map = model.compute_roa_grid(
            params,
            THETA_RANGE,
            THETA_DOT_RANGE,
            sim_duration=12.0,
            steady_state_window=4.0,
        )
        ax.pcolormesh(
            THETA_RANGE, THETA_DOT_RANGE, rolling_map, shading="auto", cmap="coolwarm"
        )
        ax.set_title(label_fn(value))
        ax.set_xlabel(r"$\theta$ (rad)")
    axes[0].set_ylabel(r"$\dot\theta$ (rad/s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename)


base_params = model.generate_params()
alpha = base_params["half_spoke_angle"]

incline_angles_deg = np.linspace(2, 40, 20)
fixed_points, floquet_multipliers = sweep_fixed_point_and_floquet(
    base_params, incline_angles_deg, with_incline
)
plot_fixed_point_and_floquet(
    incline_angles_deg,
    fixed_points,
    floquet_multipliers,
    r"incline angle $\gamma$ (deg)",
    "Effect of incline angle on the rolling gait",
    "figures/assignment_1_sweep_incline_fixed_point.png",
)

representative_inclines_deg = [5, np.rad2deg(alpha), 35]
plot_roa_panels(
    base_params,
    representative_inclines_deg,
    with_incline,
    lambda incline_deg: rf"$\gamma$ = {incline_deg:.1f}$\degree$",
    "RoA of the rolling gait as the incline steepens",
    "figures/assignment_1_sweep_incline_roa.png",
)

num_spokes_values = np.arange(6, 13)
fixed_points_n, floquet_multipliers_n = sweep_fixed_point_and_floquet(
    base_params, num_spokes_values, with_num_spokes
)
plot_fixed_point_and_floquet(
    num_spokes_values,
    fixed_points_n,
    floquet_multipliers_n,
    "number of spokes N",
    "Effect of spoke count on the rolling gait",
    "figures/assignment_1_sweep_spokes_fixed_point.png",
)

representative_num_spokes = [6, 9, 12]
plot_roa_panels(
    base_params,
    representative_num_spokes,
    with_num_spokes,
    lambda num_spokes: f"N = {num_spokes}",
    "RoA of the rolling gait vs. spoke count",
    "figures/assignment_1_sweep_spokes_roa.png",
)

plt.show()
