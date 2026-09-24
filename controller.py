"""Controllers for the inverted pendulum walker.

Two separate control problems, kept as separate functions (do one thing well):
a continuous ankle torque that balances the walker once it's near standing,
and a discrete footstep policy that chooses the landing angle of attack each
step to walk the state into that balance controller's region of attraction.
"""

import numpy as np

from integrators import rk4 as integrator
from models import inverted_pendulum_walker as model

# --- Continuous-time ankle balance controller (feedback linearization + PD) ---


def compute_ankle_torque(state, params, gains, torque_bounds):
    """Cancel the pendulum's gravity torque, then add PD feedback to upright."""
    angle, angular_velocity = state
    mass, gravity, length = params["mass"], params["gravity"], params["length"]

    cancel_gravity = -mass * gravity * length * np.sin(angle)
    stabilize = -mass * length**2 * (gains["kp"] * angle + gains["kd"] * angular_velocity)
    return np.clip(cancel_gravity + stabilize, *torque_bounds)


def simulate_standing(initial_state, params, gains, torque_bounds, sim_duration, timestep=2e-3):
    """Run the ankle-controlled pendulum (no footsteps) and return the final state."""
    state = np.array(initial_state, dtype=float)
    params = dict(params)
    n_steps = round(sim_duration / timestep)
    for step in range(n_steps):
        params["ankle_torque"] = compute_ankle_torque(state, params, gains, torque_bounds)
        state = integrator(model.dynamics, step * timestep, state, timestep, params)
    return state


def compute_roa_grid(
    theta_range, theta_dot_range, params, gains, torque_bounds, sim_duration=3.0, tolerance=1e-3
):
    """Boolean grid: does the ankle controller bring each (theta, theta_dot) to standstill?"""
    roa = np.zeros((len(theta_dot_range), len(theta_range)), dtype=bool)
    for i, theta_dot0 in enumerate(theta_dot_range):
        for j, theta0 in enumerate(theta_range):
            final_state = simulate_standing(
                [theta0, theta_dot0], params, gains, torque_bounds, sim_duration
            )
            roa[i, j] = np.linalg.norm(final_state) < tolerance
    return roa


def in_roa(state, theta_range, theta_dot_range, roa_grid):
    """Nearest-grid-point lookup: is `state` inside the precomputed RoA?"""
    theta, theta_dot = state
    i = int(np.argmin(np.abs(theta_dot_range - theta_dot)))
    j = int(np.argmin(np.abs(theta_range - theta)))
    return bool(roa_grid[i, j])


# --- Discrete-time footstep controller (Poincare section at theta = 0, mid-stance) ---


def step_to_step_map(theta_dot, angle_of_attack, params, timestep=1e-3, max_duration=5.0):
    """One stride of the Poincare map: mid-stance to the next mid-stance.

    Starting at theta = 0 with velocity `theta_dot`, walk through the touchdown
    event with the given landing angle of attack (ankle torque off), and return
    the angular velocity at the next theta = 0 crossing. Returns None if the
    walker never gets there within `max_duration` (falls over, or the swing
    never carries it back past vertical).
    """
    step_params = dict(params)
    step_params["angle_of_attack"] = angle_of_attack
    step_params["ankle_torque"] = 0.0

    state = np.array([0.0, theta_dot])
    touched_down = False
    t = 0.0
    while t < max_duration:
        next_state = integrator(model.dynamics, t, state, timestep, step_params)

        if not touched_down and model.event_guard(state, next_state, step_params):
            state = model.event_dynamics(next_state, step_params)
            touched_down = True
            t += timestep
            continue

        if touched_down and state[0] < 0.0 <= next_state[0]:
            fraction = -state[0] / (next_state[0] - state[0])
            return state[1] + fraction * (next_state[1] - state[1])

        state = next_state
        t += timestep
    return None


def nearest_index(value, grid):
    return int(np.argmin(np.abs(grid - value)))


def build_transition_table(theta_dot_grid, alpha_grid, params, timestep=1e-3):
    """next_theta_dot[i, k]: result of stepping from theta_dot_grid[i] with alpha_grid[k]."""
    next_theta_dot = np.full((len(theta_dot_grid), len(alpha_grid)), np.nan)
    for i, theta_dot in enumerate(theta_dot_grid):
        for k, alpha in enumerate(alpha_grid):
            result = step_to_step_map(theta_dot, alpha, params, timestep=timestep)
            if result is not None:
                next_theta_dot[i, k] = result
    return next_theta_dot


def solve_min_steps_policy(theta_dot_grid, next_theta_dot, already_standing):
    """Backward induction over the discretized step-to-step map.

    `already_standing[i]` marks states that need zero further steps. Returns
    (steps_to_standstill, best_alpha_index), both length len(theta_dot_grid);
    unreachable states keep steps_to_standstill == inf and best_alpha_index == -1.
    """
    n_states, n_controls = next_theta_dot.shape
    steps_to_standstill = np.where(already_standing, 0.0, np.inf)
    best_alpha_index = np.full(n_states, -1)

    changed = True
    while changed:
        changed = False
        for i in range(n_states):
            if steps_to_standstill[i] == 0:
                continue
            for k in range(n_controls):
                value = next_theta_dot[i, k]
                if np.isnan(value):
                    continue
                j = nearest_index(value, theta_dot_grid)
                candidate = steps_to_standstill[j] + 1
                if candidate < steps_to_standstill[i]:
                    steps_to_standstill[i] = candidate
                    best_alpha_index[i] = k
                    changed = True
    return steps_to_standstill, best_alpha_index
