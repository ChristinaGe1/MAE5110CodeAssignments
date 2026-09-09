# Assignment 1: Rimless Wheel

## Running the code

From the repo root, with the `uv` environment synced (`uv sync --python 3.14`):

```console
uv run assignment_1_sanity_check.py
uv run assignment_1_regions_of_attraction.py
uv run assignment_1_return_map.py
uv run assignment_1_floquet.py
uv run assignment_1_sweep.py
```

The model itself lives in `models/rimless_wheel.py` (dynamics, guard, reset map, and
the hybrid `simulate` function); `integrators.py` provides the RK4 step it's built on.

## Model

State: $[\theta, \dot\theta]$, $\theta$ = stance-spoke angle from vertical, positive downhill.

- Dynamics (inverted pendulum about the stance pivot): $\ddot\theta = \frac{g}{l}\sin\theta$.
  $\gamma$ doesn't appear since $\theta$ is measured from gravity's fixed direction.
- Guard (leading spoke touches down): $\theta = \gamma + \alpha$, $\alpha = \pi/N$ (half-spoke angle).
- Reset (plastic collision, relabels stance spoke): $\theta^+ = \theta^- - 2\alpha$,
  $\dot\theta^+ = \dot\theta^-\cos(2\alpha)$.

## Sanity checks

Three checks, in `assignment_1_sanity_check.py`:

**1. Energy conservation within a stance phase.** No damping/actuation between impacts,
so mechanical energy should stay flat. Result: max drift $6\times10^{-14}$ J/kg —
conserved to numerical precision.

![Energy conservation within a stance phase](figures/assignment_1_sanity_check_1_energy.png)

**2. Impact energy loss matches theory.** Theory predicts a fixed ratio
$KE^+/KE^- = \cos^2(2\alpha)$ at every impact. Result: with 8 spokes ($\alpha = \pi/8$),
measured ratio is exactly 0.5, matching $\cos^2(\pi/4)$.

**3. Sustained rolling converges to a steady impact speed.** Expected the post-impact
velocity to settle to a constant (the rolling limit cycle). Result: converges to
$\dot\theta^* \approx 1.6148$ rad/s within a handful of impacts.

![Sustained rolling: theta(t) sawtoothing between resets and the guard](figures/assignment_1_sanity_check_3_rolling.png)

## Regions of attraction

Grid over $(\theta, \dot\theta) \in [-1.5, 1.5]\times[-6,6]$ (61×61), simulating 20 s
from each point and checking whether impacts are still occurring near the end.

![Region of attraction of the rolling gait](figures/assignment_1_roa.png)

Expected exactly **one** bounded attractor: the rolling limit cycle (falling over or
rocking forever aren't stable attracting sets). The plot confirms this: a red basin
converging to steady rolling, and a blue wedge that doesn't. The wedge boundary is the
pendulum's separatrix at $\theta=0$ — trajectories inside lack the energy to swing past
it and reach the guard, so they oscillate forever without stepping.

**Efficiency note:** only the impact needs precise crossing time; the smooth swing
doesn't. So `simulate` takes RK4 steps of a few ms during the swing and only bisects
the last step at the guard crossing, instead of using one tiny fixed step everywhere.
That took the 3,721-point grid (20 s each) to about 3.5 minutes.

## Return map and Floquet multiplier

Poincaré section: the impact event. `assignment_1_return_map.py` sweeps a range of
post-impact velocities $\dot\theta_n$ and simulates one stance phase from each to get
$\dot\theta_{n+1}$.

![Return map with identity line and fixed point](figures/assignment_1_return_map.png)

The map crosses the identity line at $\dot\theta^* \approx 1.6148$ rad/s, matching the
sanity check. `assignment_1_floquet.py` estimates the local slope by perturbing
$\pm 10^{-4}$ rad/s around that fixed point:

$$\text{Floquet multiplier} = \frac{f(\dot\theta^*+\epsilon) - f(\dot\theta^*-\epsilon)}{2\epsilon} \approx 0.5$$

Since $|0.5| < 1$, the limit cycle is locally stable — consistent with the RoA plot.

## Effect of incline angle and spoke count

`assignment_1_sweep.py` sweeps both parameters.

![Fixed point and Floquet multiplier vs. incline angle](figures/assignment_1_sweep_incline_fixed_point.png)
![RoA as the incline steepens](figures/assignment_1_sweep_incline_roa.png)

**Incline angle $\gamma$:** steeper slopes give faster steady rolling (more PE released
per step). The Floquet multiplier stays flat at 0.5 across the sweep — it depends only
on spoke geometry ($\cos^2(2\alpha)$), not $\gamma$. The RoA basin also changes shape:
shallow angles leave a large non-rolling wedge from the $\theta=0$ barrier; once
$\gamma$ exceeds $\alpha$ the barrier disappears and rolling covers nearly the whole
swept space.

![Fixed point and Floquet multiplier vs. spoke count](figures/assignment_1_sweep_spokes_fixed_point.png)
![RoA vs. spoke count](figures/assignment_1_sweep_spokes_roa.png)

**Number of spokes $N$:** the Floquet multiplier rises with $N$ — 0.5 at $N=8$ to 0.75
at $N=12$ — matching $\cos^2(2\pi/N)$, independent of incline. Fewer spokes → bigger
$\alpha$ → harder collisions → stronger contraction (faster convergence, more energy
lost per step); more spokes → gentler collisions, slower convergence, approaching the
lossless limit as $N\to\infty$.

More strikingly, **$N=6$ has no rolling gait at the default 10° incline** — its RoA
panel is entirely blue, and `find_fixed_point` finds nothing. With $\alpha=30°$, the
energy needed to clear the $\theta=0$ barrier exceeds what the fixed point would
supply — the wheel needs more spokes or a steeper slope to roll at all. A real
threshold effect: $N=8$–$12$ sustain rolling here; $N=6,7$ don't.
