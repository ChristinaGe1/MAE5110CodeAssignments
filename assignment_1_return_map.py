import matplotlib.pyplot as plt
import numpy as np

from models import rimless_wheel as model

params = model.generate_params()
alpha = params["half_spoke_angle"]

v_n = np.linspace(0.8, 4.0, 60)
v_next = np.array([model.poincare_map(v, params) for v in v_n])

fixed_point = model.find_fixed_point(params)
print(f"Fixed point of the return map: v* = {fixed_point:.6f} rad/s")

initial_state = np.array([model.touchdown_angle(params) - 2 * alpha, 0.85])
result = model.simulate(initial_state, params, sim_duration=15.0)
trajectory_velocities = result["post_impact_velocities"]

plot_min, plot_max = 0.0, v_n.max()

plt.figure()
plt.plot([plot_min, plot_max], [plot_min, plot_max], "k--", label="identity")
plt.plot(v_n, v_next, "-", label="return map " + r"$f(\dot\theta_n)$")
plt.plot(
    trajectory_velocities[:-1],
    trajectory_velocities[1:],
    "o",
    color="tab:orange",
    label="one trajectory's impacts",
)
plt.plot(
    fixed_point,
    fixed_point,
    "r*",
    markersize=15,
    label=f"fixed point ({fixed_point:.3f})",
)
plt.xlabel(r"$\dot\theta_n$ (rad/s)")
plt.ylabel(r"$\dot\theta_{n+1}$ (rad/s)")
plt.title("Rimless wheel: step-to-step return map")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.savefig("figures/assignment_1_return_map.png")
plt.show()
