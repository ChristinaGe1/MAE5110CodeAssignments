import numpy as np

from integrators import rk4
from models import pendulum


def test_free_pendulum_energy():
    NUM_TIMESTEPS = 100
    TIMESTEP = 0.01

    params = pendulum.generate_params()
    params["torque"] = 0.0
    params["damping_coeff"] = 0.0

    state = np.array([np.pi - np.pi / 4, 0.1])
    t = 0.0
    state_traj = np.zeros((2, NUM_TIMESTEPS))

    for i in range(NUM_TIMESTEPS):
        state_traj[:, i] = state
        state = rk4(pendulum.dynamics, t, state, TIMESTEP, params)
        t += TIMESTEP

    kinetic, potential = pendulum.calculate_energy(state_traj, params)
    total = kinetic + potential

    assert np.allclose(total, total[0])


def test_pendulum_torque():
    NUM_TIMESTEPS = 100
    TIMESTEP = 0.1

    initial_state = np.array([np.pi - np.pi / 4, 0.0])

    params = pendulum.generate_params()
    # change params so they're not defaults which may hide bugs
    params["mass"] = 2.0
    params["length"] = 3.0
    params["gravity"] = 4.0

    params["torque"] = (
        -params["mass"]
        * params["gravity"]
        * params["length"]
        * np.sin(initial_state[0])
    )

    state = initial_state
    t = 0.0
    state_traj = np.zeros((2, NUM_TIMESTEPS))

    for i in range(NUM_TIMESTEPS):
        state_traj[:, i] = state
        state = rk4(pendulum.dynamics, t, state, TIMESTEP, params)
        t += TIMESTEP

    assert np.allclose(state_traj, initial_state.reshape(2, 1))


def test_pendulum_damping():
    NUM_TIMESTEPS = 15000
    TIMESTEP = 1e-2

    initial_state = np.array([np.pi - np.pi / 4, 0.0])

    params = pendulum.generate_params()
    params["damping"] = 1.0
    params["torque"] = 0.0

    state = initial_state
    t = 0.0
    state_traj = np.zeros((2, NUM_TIMESTEPS))

    for i in range(NUM_TIMESTEPS):
        state_traj[:, i] = state
        state = rk4(pendulum.dynamics, t, state, TIMESTEP, params)
        t += TIMESTEP

    kinetic, potential = pendulum.calculate_energy(state_traj, params)
    total = kinetic + potential

    assert np.all(np.diff(total) <= 0.0)
    assert np.allclose(state_traj[0, -1], np.pi, rtol=1e-3)
