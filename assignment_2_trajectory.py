"""Full trajectory for an initial condition that needs at least 3 footsteps.

Runs the walker under the footstep lookup table (assignment_2_lookup_table.py)
until it enters the ankle controller's region of attraction (RoA), switches to
the ankle balance controller, and settles at standstill. Also estimates, for
the same initial speed, the most steps a valid (if suboptimal) sequence of
footsteps could take before being forced into the RoA -- see the docstring on
`find_longest_path` for exactly what that number means.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import controller
from integrators import rk4 as integrator
from models import inverted_pendulum_walker as model

params = model.generate_params()
mass, gravity, length = params["mass"], params["gravity"], params["length"]
torque_bounds = (-0.1 * mass * gravity * length, 0.05 * mass * gravity * length)
gains = {"kp": 20.0, "kd": 8.0}

roa_data = np.load("output/assignment_2/roa_grid.npz")
theta_range = roa_data["theta_range"]
theta_dot_range = roa_data["theta_dot_range"]
roa_grid = roa_data["roa_grid"]

table_data = np.load("output/assignment_2/lookup_table.npz")
theta_dot_grid = table_data["theta_dot_grid"]
steps_to_standstill = table_data["steps_to_standstill"]
best_alpha_index = table_data["best_alpha_index"]
alpha_grid = table_data["alpha_grid"]
transition_table = table_data["transition_table"]
already_standing = table_data["already_standing"]


def choose_alpha(theta_dot):
    index = controller.nearest_index(theta_dot, theta_dot_grid)
    control_index = best_alpha_index[index]
    return alpha_grid[control_index] if control_index >= 0 else alpha_grid[0]


def simulate_trajectory(initial_theta_dot, timestep=1e-3, max_time=30.0):
    """Walk under the lookup-table policy, then stand once inside the RoA."""
    state = np.array([0.0, initial_theta_dot])
    sim_params = dict(params)
    standing = False
    step_count = 0
    at_midstance = True

    t = 0.0
    time_traj = [t]
    state_traj = [state.copy()]

    while t < max_time:
        if at_midstance and not standing:
            if controller.in_roa(state, theta_range, theta_dot_range, roa_grid):
                standing = True
            else:
                sim_params["angle_of_attack"] = choose_alpha(state[1])
            at_midstance = False

        sim_params["ankle_torque"] = (
            controller.compute_ankle_torque(state, sim_params, gains, torque_bounds)
            if standing
            else 0.0
        )
        next_state = integrator(model.dynamics, t, state, timestep, sim_params)

        if not standing and model.event_guard(state, next_state, sim_params):
            next_state = model.event_dynamics(next_state, sim_params)
            step_count += 1
        elif not standing and state[0] < 0.0 <= next_state[0]:
            at_midstance = True

        t += timestep
        state = next_state
        time_traj.append(t)
        state_traj.append(state.copy())

        if standing and np.linalg.norm(state) < 1e-3:
            break

    return np.array(time_traj), np.array(state_traj).T, step_count


def find_longest_path(start_index, max_depth, max_nodes=300_000):
    """Longest simple path (no repeated grid states) from `start_index` to any
    already-standing state, using any action at each state -- not just the
    optimal policy. This answers "what is the most steps a *valid* sequence of
    footsteps could take before being forced into the RoA", under the
    assumption that the walker never deliberately revisits the same discretized
    speed twice (revisiting would let it stall indefinitely, which is a
    resolution artifact rather than a meaningful gait).
    """
    best_depth = 0
    nodes_visited = 0

    def visit(index, visited, depth):
        nonlocal best_depth, nodes_visited
        nodes_visited += 1
        if nodes_visited > max_nodes or depth >= max_depth:
            return
        for control_index in range(transition_table.shape[1]):
            next_speed = transition_table[index, control_index]
            if np.isnan(next_speed):
                continue
            next_index = controller.nearest_index(next_speed, theta_dot_grid)
            if next_index in visited:
                continue
            new_depth = depth + 1
            if already_standing[next_index]:
                best_depth = max(best_depth, new_depth)
                continue
            visit(next_index, visited | {next_index}, new_depth)

    visit(start_index, {start_index}, 0)
    return best_depth


# --- Pick an initial condition that needs at least 3 steps -------------------
candidates = np.where(np.isfinite(steps_to_standstill) & (steps_to_standstill >= 3))[0]
start_index = candidates[0]
initial_theta_dot = theta_dot_grid[start_index]
print(f"Initial mid-stance speed: {initial_theta_dot:.3f} rad/s")
print(f"Minimum steps to standstill (lookup table): {int(steps_to_standstill[start_index])}")

max_depth = 2 * int(steps_to_standstill[start_index]) + 4
longest = find_longest_path(start_index, max_depth)
print(f"Longest valid non-repeating footstep sequence before reaching the RoA: {longest} steps")

time_traj, state_traj, step_count = simulate_trajectory(initial_theta_dot)
print(f"Simulated trajectory took {step_count} footsteps to reach standstill")

output = Path("output/assignment_2")
output.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 6), layout="constrained")
axes[0].plot(time_traj, state_traj[0])
axes[0].set_ylabel(r"$\theta$ (rad)")
axes[0].set_title(
    f"Trajectory from $\\dot\\theta_0$ = {initial_theta_dot:.3f} rad/s "
    f"({step_count} steps to standstill)"
)
axes[1].plot(time_traj, state_traj[1])
axes[1].set_ylabel(r"$\dot\theta$ (rad/s)")
axes[1].set_xlabel("time (s)")
plt.savefig(output / "trajectory.png")
plt.show()
