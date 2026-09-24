"""Footstep lookup table: how many steps to standstill, from any initial speed.

Builds the discrete step-to-step map (Poincare section at theta = 0) over a
grid of mid-stance angular velocities and landing angles of attack, then backs
out, by induction from the ankle controller's region of attraction (RoA), the
minimum number of steps needed to reach standstill from each grid state and
the angle of attack that achieves it.

Also checks grid resolution: the minimum-steps count for a handful of test
initial conditions is compared across increasingly fine grids until it stops
changing, to justify the coarsest resolution that is still accurate.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import controller
from models import inverted_pendulum_walker as model

params = model.generate_params()
gravity, length = params["gravity"], params["length"]

froude_velocity = np.sqrt(2 * gravity / length)  # Froude number 2 cutoff
alpha_min, alpha_max = np.pi / 8, np.pi / 7
alpha_grid = np.linspace(alpha_min, alpha_max, 8)

roa_data = np.load("output/assignment_2/roa_grid.npz")
theta_range = roa_data["theta_range"]
theta_dot_range = roa_data["theta_dot_range"]
roa_grid = roa_data["roa_grid"]


def solve_for_resolution(n_states):
    theta_dot_grid = np.linspace(0.0, froude_velocity, n_states)
    transition_table = controller.build_transition_table(theta_dot_grid, alpha_grid, params)
    already_standing = np.array(
        [controller.in_roa((0.0, td), theta_range, theta_dot_range, roa_grid) for td in theta_dot_grid]
    )
    steps, best_alpha_index = controller.solve_min_steps_policy(
        theta_dot_grid, transition_table, already_standing
    )
    return theta_dot_grid, transition_table, already_standing, steps, best_alpha_index


# --- Grid resolution check ---------------------------------------------------
# Compare the minimum-steps count for a fixed set of test speeds across
# increasingly fine grids. A resolution is converged once its answer matches
# every finer grid; a match with only the next grid can be coincidental.
test_speeds = np.linspace(0.5, 0.9 * froude_velocity, 6)
resolutions = [10, 20, 40, 80, 160]

print("Grid resolution check (steps-to-standstill at fixed test speeds):")
solutions = {}
steps_at_test = {}
for n_states in resolutions:
    solutions[n_states] = solve_for_resolution(n_states)
    theta_dot_grid, _, _, steps, _ = solutions[n_states]
    steps_at_test[n_states] = np.array(
        [steps[controller.nearest_index(v, theta_dot_grid)] for v in test_speeds]
    )
    print(f"  n_states={n_states:4d}: {steps_at_test[n_states]}")

final_n_states = next(
    n
    for i, n in enumerate(resolutions)
    if all(np.array_equal(steps_at_test[n], steps_at_test[finer]) for finer in resolutions[i + 1 :])
)
print(f"Coarsest converged grid: n_states={final_n_states}")

theta_dot_grid, transition_table, already_standing, steps_to_standstill, best_alpha_index = solutions[
    final_n_states
]

output = Path("output/assignment_2")
output.mkdir(parents=True, exist_ok=True)
np.savez(
    output / "lookup_table.npz",
    theta_dot_grid=theta_dot_grid,
    steps_to_standstill=steps_to_standstill,
    best_alpha_index=best_alpha_index,
    alpha_grid=alpha_grid,
    transition_table=transition_table,
    already_standing=already_standing,
)

reachable = np.isfinite(steps_to_standstill)
print(
    f"\nFinal grid (n_states={final_n_states}): "
    f"{reachable.mean():.1%} of states reach standstill in a finite number of steps, "
    f"up to {int(steps_to_standstill[reachable].max())} steps."
)

plt.figure()
plt.plot(theta_dot_grid[reachable], steps_to_standstill[reachable], "o-")
plt.plot(
    theta_dot_grid[~reachable],
    np.zeros(np.count_nonzero(~reachable)),
    "rx",
    label="never reaches standstill",
)
plt.xlabel(r"initial mid-stance speed $\dot\theta_0$ (rad/s)")
plt.ylabel("steps to standstill")
plt.title("Minimum steps to standstill vs. initial condition")
plt.legend()
plt.tight_layout()
plt.savefig(output / "steps_to_standstill.png")
plt.show()
