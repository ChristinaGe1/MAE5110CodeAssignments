import time as timer

import matplotlib.pyplot as plt
import numpy as np

from models import rimless_wheel as model

params = model.generate_params()

theta_range = np.linspace(-1.5, 1.5, 61)
theta_dot_range = np.linspace(-6.0, 6.0, 61)

sim_duration = 20.0
steady_state_window = 5.0

start_time = timer.time()
rolling_map = model.compute_roa_grid(
    params, theta_range, theta_dot_range, sim_duration, steady_state_window
)
elapsed = timer.time() - start_time
print(f"Grid of {rolling_map.size} points took {elapsed:.1f} s")

plt.figure()
plt.pcolormesh(
    theta_range, theta_dot_range, rolling_map, shading="auto", cmap="coolwarm"
)
plt.xlabel(r"$\theta$ (rad)")
plt.ylabel(r"$\dot\theta$ (rad/s)")
plt.title("Region of attraction of the rolling limit cycle")
plt.colorbar(label="1 = converges to rolling gait, 0 = does not")
plt.tight_layout()
plt.savefig("figures/assignment_1_roa.png")
plt.show()
