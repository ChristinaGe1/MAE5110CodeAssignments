"""Region of attraction (RoA) of the ankle balance controller.

Grid-searches (theta, theta_dot) initial conditions and checks which ones the
feedback-linearizing PD ankle controller (controller.compute_ankle_torque)
brings to standstill, given the torque is bounded to a small fraction of mgl.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import controller
from models import inverted_pendulum_walker as model

params = model.generate_params()
mass, gravity, length = params["mass"], params["gravity"], params["length"]

# Bounds from the assignment: tau in [-0.1 mgl, 0.05 mgl].
torque_bounds = (-0.1 * mass * gravity * length, 0.05 * mass * gravity * length)
gains = {"kp": 20.0, "kd": 8.0}

# The torque budget alone only cancels gravity up to about |sin(theta)| <~ 0.1,
# so a small grid around the upright equilibrium is enough to find the RoA.
theta_range = np.linspace(-0.2, 0.2, 41)
theta_dot_range = np.linspace(-1.5, 1.5, 41)

start = time.time()
roa_grid = controller.compute_roa_grid(theta_range, theta_dot_range, params, gains, torque_bounds)
print(f"Grid of {roa_grid.size} points took {time.time() - start:.1f} s")
print(f"RoA covers {roa_grid.mean():.1%} of the grid")

output = Path("output/assignment_2")
output.mkdir(parents=True, exist_ok=True)
np.savez(
    output / "roa_grid.npz",
    theta_range=theta_range,
    theta_dot_range=theta_dot_range,
    roa_grid=roa_grid,
    torque_bounds=torque_bounds,
    gains_kp=gains["kp"],
    gains_kd=gains["kd"],
)

plt.figure()
plt.pcolormesh(theta_range, theta_dot_range, roa_grid, shading="auto", cmap="coolwarm")
plt.xlabel(r"$\theta$ (rad)")
plt.ylabel(r"$\dot\theta$ (rad/s)")
plt.title("Region of attraction of the ankle balance controller")
plt.colorbar(label="1 = converges to standstill, 0 = does not")
plt.tight_layout()
plt.savefig(output / "roa.png")
plt.show()
