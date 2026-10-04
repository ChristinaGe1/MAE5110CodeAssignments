
## Running the code

```console
uv run assignment_2_roa.py
uv run assignment_2_lookup_table.py
uv run assignment_2_trajectory.py
```

Each script uses the output of the one before it, so run them in this order. The controllers are
in `controller.py`.

## Sketches

![Walker at mid-stance, before and after touchdown, and falling, with each one marked on the state space](output/assignment_2/sktech.jpg)

## Ankle controller RoA

The ankle controller cancels out gravity, then adds a spring and damper that pull the pendulum
back to upright. I used gains of 20 and 8, and the torque is clipped to the bounds from the
assignment.

To find the RoA, I simulated a 41 × 41 grid of starting angles and speeds around upright. A state
counts as inside the RoA if the walker is basically standing still after 3 seconds.

![Region of attraction of the ankle controller](output/assignment_2/roa.png)

When the leg is straight up, the controller can catch the walker at speeds from -0.15 to 0.30 rad/s.

## Poincaré section

I put the section at mid-stance, where the stance leg is straight up.
Since theta is fixed at this section, angular velocity is the only state needed for the return map. This also avoids using touchdown as the section because the touchdown angle changes with the selected landing angle.

## Grid resolution

I picked 6 test speeds and checked how many steps each one needs to stop on grids of different
sizes. I counted a grid as good enough when it matched every finer grid.

| states | 0.50 | 1.20 | 1.89 | 2.59 | 3.29 | 3.99 |
|---:|:-:|:-:|:-:|:-:|:-:|:-:|
| 10  | 1 | 1     | 2 | 3 | 3 | 4     |
| 20  | 1 | 1     | 2 | 3 | 3 | **3** |
| 40  | 1 | **2** | 2 | 3 | 3 | 4     |
| 80  | 1 | 1     | 2 | 3 | 3 | 4     |
| 160 | 1 | 1     | 2 | 3 | 3 | 4     |

I went with 80 states. At 40, the answer at 1.20 rad/s is wrong, so 40 is too coarse. The 10-state
grid matches too, but since 20 and 40 don't, I think that's just luck.

## Steps to standstill

![Minimum steps to standstill vs. initial speed](output/assignment_2/steps_to_standstill.png)

| steps | starting speed (rad/s) |
|:-:|:-:|
| 0 | 0 – 0.34 |
| 1 | 0.45 – 1.23 |
| 2 | 1.29 – 2.30 |
| 3 | 2.35 – 3.87 |
| 4 | 3.92 – 4.43 |

79 of the 80 states make it to standstill. The only one that doesn't is at 0.39 rad/s (the red ×).

## Trajectory (at least 3 steps)

I started the walker at 2.355 rad/s. Following the lookup table, it stops in 3 steps. The most it
can keep walking before it reaches the RoA is 5 steps. I found that by searching for the longest
footstep sequence that never lands on the same grid speed twice.

![Optimal policy vs. longest path](output/assignment_2/trajectory.png)
