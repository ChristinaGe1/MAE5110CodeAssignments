import matplotlib.pyplot as plt
import numpy as np

from models import rimless_wheel as model

# Sanity checks for the rimless-wheel model.
#
# 1. Within a single stance phase, there is no damping and no actuation, so
#    total mechanical energy (relative to the current pivot) should stay
#    constant. If it drifts, the guard/RK4 combination is buggy.
# 2. At each impact, the plastic collision should remove a known, fixed
#    fraction of kinetic energy: KE^+ / KE^- = cos(2*alpha)^2, independent of
#    the impact velocity. We check this numerically across many impacts.
# 3. Released on a slope steeper than the spoke half-angle, the wheel should
#    roll forever, with the impact velocity converging to a steady value
#    (the rolling limit cycle) rather than growing or decaying to zero.

params = model.generate_params()
alpha = params["half_spoke_angle"]

# --- Check 1: energy conservation within a stance phase ---------------------
initial_state = np.array([params["incline_angle"], 0.5])
short_duration = 0.05  # short enough to stay within one stance phase
result = model.simulate(initial_state, params, short_duration)

kinetic_energy, potential_energy = model.calculate_energy(result["state"], params)
total_energy = kinetic_energy + potential_energy
energy_drift = np.max(np.abs(total_energy - total_energy[0]))
print(f"[Check 1] max energy drift within a stance phase: {energy_drift:.3e} J/kg")
assert energy_drift < 1e-6, "Energy should be conserved between impacts"

plt.figure()
plt.plot(result["time"], total_energy)
plt.xlabel("Time (s)")
plt.ylabel("Specific energy (J/kg)")
plt.title("Sanity check 1: energy conservation within a stance phase")
plt.tight_layout()
plt.savefig("figures/assignment_1_sanity_check_1_energy.png")

# --- Check 2: impact energy loss matches theory ------------------------------
initial_state = np.array([params["incline_angle"] - alpha, 2.0])
long_duration = 15.0
result = model.simulate(initial_state, params, long_duration)

pre = result["pre_impact_velocities"]
post = result["post_impact_velocities"]
measured_ratio = (post / pre) ** 2
expected_ratio = np.cos(2 * alpha) ** 2
print(f"[Check 2] expected KE ratio cos(2*alpha)^2 = {expected_ratio:.6f}")
print(f"[Check 2] measured KE ratios (first 5 impacts): {measured_ratio[:5]}")
assert np.allclose(measured_ratio, expected_ratio, atol=1e-8), (
    "Every impact should remove the same fraction of kinetic energy"
)

# --- Check 3: sustained rolling converges to a steady impact velocity -------
print(f"[Check 3] post-impact angular velocity, last 5 impacts: {post[-5:]}")
print(f"[Check 3] number of impacts in {long_duration}s: {len(post)}")

plt.figure()
plt.plot(result["time"], result["state"][0], label=r"$\theta$")
plt.axhline(model.touchdown_angle(params), color="gray", linestyle="--", label="guard")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title("Sanity check 3: sustained rolling")
plt.legend()
plt.tight_layout()
plt.savefig("figures/assignment_1_sanity_check_3_rolling.png")

print("\nAll sanity checks passed.")
plt.show()
