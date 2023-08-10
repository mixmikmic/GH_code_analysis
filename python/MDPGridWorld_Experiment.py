import math
import numpy as np
from MDPGridWorld import *

np.random.seed(0)

book_grid = [[' ',' ',' ',+1],
            [' ','#',' ',-1],
            ['@',' ',' ',' ']]

gw = MDPGridWorld(book_grid, action_noise_dist=[0.1, 0.8, 0.1], obstacle_reward=-2, verbose=True)

gw.curr_state_idx, _ = gw.sample_next_state(gw.curr_state_idx, MDPGridWorld.North)
print(gw)

print("States (fused into a single grid-bc it's possible to do so here):")
gw.disp_custom_grid(range(gw.nS), formatting=lambda x: str(x))

vi = run_value_iteration(gw.T, gw.R, gw.gamma)
print("Optimal Value:")
gw.disp_custom_grid(vi.V, formatting=lambda x: "{:.3f}".format(x))
print("\nOptimal Policy:")
gw.disp_custom_grid(vi.policy, lambda x: "{:}".format(gw.actions_name[x]))

tau = gw.sample_trajectory(init_state_idx="random", max_length=5)
print(tau)
gw.interpret_trajectory(tau)

tau_list = gw.sample_trajectories(10, max_length=10)
for i, tau in enumerate(tau_list): print("T{:03d}: {}".format(i, tau))

gw = MDPGridWorld(book_grid, action_noise_dist=[0.4, 0.2, 0.4], obstacle_reward=-0.2, visit_obstacles=True, verbose=True)

gw.disp_policy(gw._get_optimal_policy())

tau_list = gw.sample_trajectories(10, init_state_idx=6, max_length=10)
for i, tau in enumerate(tau_list): print("T{:03d}: {}".format(i, tau))

# This trajectory visits the obstacle
gw.interpret_trajectory(tau_list[8])

gw = MDPGridWorld(book_grid, action_noise_dist=[0.4, 0.2, 0.4], obstacle_reward=-10, visit_obstacles=True, verbose=True)

gw.disp_policy(gw._get_optimal_policy())

tau_list = gw.sample_trajectories(10, init_state_idx=6, max_length=10)
for i, tau in enumerate(tau_list): print("T{:03d}: {}".format(i, tau))

