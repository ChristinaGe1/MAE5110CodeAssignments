import numpy as np

from integrators import rk4
from models import pendulum


def test_free_pendulum_energy():
    NUM_TIMESTEPS = 100
    TIMESTEP = 0.01

    params = pendulum.generate_params()
    params["torque"] = 0.0
    params["damping_coeff"] = 0.0

    state = np.array([np.pi / 4, 0.1])
    t = 0.0
    state_traj = np.zeros((2, NUM_TIMESTEPS))

    for i in range(NUM_TIMESTEPS):
        state_traj[:, i] = state
        state = rk4(pendulum.dynamics, t, state, TIMESTEP, params)
        t += TIMESTEP

    kinetic, potential = pendulum.calculate_energy(state_traj, params)
    total = kinetic + potential

    assert np.allclose(total, total[0])
