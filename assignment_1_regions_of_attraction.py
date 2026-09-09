import time as timer

import matplotlib.pyplot as plt
import numpy as np

from models import rimless_wheel as model

# Brute-force regions of attraction (RoA).
#
# The rimless wheel, as modeled here (a single, one-directional guard for the
# leading spoke), has exactly one bounded attractor: the steady rolling limit
# cycle. Everything else either (a) oscillates forever without ever reaching
# the guard (not enough energy to swing past the unstable equilibrium at
# theta=0), or (b) runs away with theta -> -infinity (falls over backward).
# Neither of those is a bounded attracting set, so we classify the grid into
# just two classes: "converges to the rolling gait" vs. "does not".
#
# Speed trick for the hint: instead of using a globally tiny fixed timestep
# (needed only to resolve the impact instant precisely), `model.simulate`
# already uses a moderate fixed RK4 step for the smooth swing and only
# refines the timestep locally, via bisection, right at the guard crossing.
# That is what makes a 61x61 grid of 20-second simulations tractable here.

params = model.generate_params()

theta_range = np.linspace(-1.5, 1.5, 61)
theta_dot_range = np.linspace(-6.0, 6.0, 61)

sim_duration = 20.0
steady_state_window = 5.0  # classify as "rolling" if it's still impacting near the end

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
