import numpy as np

from integrators import rk4 as integrator


def generate_params():
    num_spokes = 8
    params = {
        "gravity": 9.81,
        "spoke_length": 1.0,
        "num_spokes": num_spokes,
        "half_spoke_angle": np.pi / num_spokes,
        "incline_angle": np.deg2rad(10),
    }
    return params


def dynamics(t, state, params):
    gravity = params["gravity"]
    spoke_length = params["spoke_length"]

    angle = state[0]
    angular_velocity = state[1]

    angular_acceleration = (gravity / spoke_length) * np.sin(angle)

    state_derivative = np.array([angular_velocity, angular_acceleration])
    return state_derivative


def touchdown_angle(params):
    return params["incline_angle"] + params["half_spoke_angle"]


def detect_event(state, params):
    angle = state[0]
    return angle - touchdown_angle(params)


def reset_state(state, params):
    """Plastic collision: relabel the new stance spoke and update velocity."""
    angle, angular_velocity = state
    alpha = params["half_spoke_angle"]

    new_angle = angle - 2 * alpha
    new_angular_velocity = angular_velocity * np.cos(2 * alpha)
    return np.array([new_angle, new_angular_velocity])


def calculate_energy(state, params):
    """Kinetic and potential energy per unit mass, relative to the current stance pivot."""
    gravity = params["gravity"]
    spoke_length = params["spoke_length"]

    angle = state[0]
    angular_velocity = state[1]

    kinetic_energy = 0.5 * (spoke_length * angular_velocity) ** 2
    potential_energy = gravity * spoke_length * np.cos(angle)
    return kinetic_energy, potential_energy


def find_impact(t0, state0, timestep, params, tolerance=1e-10, max_iterations=60):
    """Bisect within [t0, t0 + timestep] to locate the impact time precisely."""
    low, high = 0.0, timestep
    impact_state = state0
    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        impact_state = integrator(dynamics, t0, state0, mid, params)
        if detect_event(impact_state, params) < 0:
            low = mid
        else:
            high = mid
        if high - low < tolerance:
            break
    impact_time = t0 + high
    impact_state = integrator(dynamics, t0, state0, high, params)
    return impact_time, impact_state


def simulate(initial_state, params, sim_duration, max_timestep=2e-3):
    """Simulate the hybrid rimless-wheel dynamics for `sim_duration` seconds."""
    time = 0.0
    state = np.asarray(initial_state, dtype=float)

    time_trajectory = [time]
    state_trajectory = [state.copy()]
    impact_times = []
    pre_impact_velocities = []
    post_impact_velocities = []

    while time < sim_duration:
        timestep = min(max_timestep, sim_duration - time)
        next_state = integrator(dynamics, time, state, timestep, params)

        if detect_event(state, params) < 0 and detect_event(next_state, params) >= 0:
            impact_time, impact_state = find_impact(time, state, timestep, params)

            time_trajectory.append(impact_time)
            state_trajectory.append(impact_state.copy())
            pre_impact_velocities.append(impact_state[1])

            state = reset_state(impact_state, params)
            time = impact_time

            impact_times.append(impact_time)
            post_impact_velocities.append(state[1])

            time_trajectory.append(time)
            state_trajectory.append(state.copy())
        else:
            time += timestep
            state = next_state
            time_trajectory.append(time)
            state_trajectory.append(state.copy())

    result = {
        "time": np.array(time_trajectory),
        "state": np.array(state_trajectory).T,  # shape (2, N)
        "impact_times": np.array(impact_times),
        "pre_impact_velocities": np.array(pre_impact_velocities),
        "post_impact_velocities": np.array(post_impact_velocities),
    }
    return result


def poincare_map(post_impact_velocity, params, max_duration=10.0):
    """Step-to-step return map. Returns None if the guard is never reached."""
    alpha = params["half_spoke_angle"]
    initial_state = np.array(
        [touchdown_angle(params) - 2 * alpha, post_impact_velocity]
    )
    result = simulate(initial_state, params, max_duration)
    if len(result["post_impact_velocities"]) == 0:
        return None
    return result["post_impact_velocities"][0]


def find_fixed_point(params, bracket=(0.8, 10.0), tolerance=1e-10, max_iterations=100):
    """Bisection search for the fixed point of the Poincare return map."""

    def residual(velocity):
        mapped_velocity = poincare_map(velocity, params)
        if mapped_velocity is None:
            raise RuntimeError(f"velocity {velocity} never reaches the guard")
        return mapped_velocity - velocity

    low, high = bracket
    low_residual = residual(low)
    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        mid_residual = residual(mid)
        if np.sign(mid_residual) == np.sign(low_residual):
            low, low_residual = mid, mid_residual
        else:
            high = mid
        if high - low < tolerance:
            break
    return 0.5 * (low + high)


def compute_roa_grid(
    params, theta_range, theta_dot_range, sim_duration=20.0, steady_state_window=5.0
):
    """Brute-force region-of-attraction grid for the rolling limit cycle."""
    rolling_map = np.zeros((len(theta_dot_range), len(theta_range)), dtype=bool)
    for i, theta_dot0 in enumerate(theta_dot_range):
        for j, theta0 in enumerate(theta_range):
            result = simulate(np.array([theta0, theta_dot0]), params, sim_duration)
            impact_times = result["impact_times"]
            rolling_map[i, j] = (
                len(impact_times) > 0
                and impact_times[-1] > sim_duration - steady_state_window
            )
    return rolling_map
