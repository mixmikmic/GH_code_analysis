prob = np.zeros((2, 5, 5))

prob[0] = [[0.3, 0.7, 0., 0., 0.],
           [0.3, 0.0, 0.7, 0., 0.],
           [0.3, 0.0, 0., 0.7, 0.],
           [0.3, 0.0, 0., 0., 0.7],
           [0.3, 0.0, 0., 0., 0.7]]

prob[1] = [[1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.]]

rewards = np.zeros((5, 2))
rewards[0] = [0., 0.]
rewards[1] = [0., 1.]
rewards[2] = [0., 1.]
rewards[3] = [0., 1.]
rewards[4] = [0.3, 2.]

vi = mdptoolbox.mdp.ValueIteration(prob, rewards, 0.9)
vi.run()

optimal_policy = vi.policy
expected_values = vi.V

import mdptoolbox
import numpy as np

prob = np.zeros((2, 5, 5))

prob[0] = [[0.3, 0.7, 0., 0., 0.],
           [0.3, 0.0, 0.7, 0., 0.],
           [0.3, 0.0, 0., 0.7, 0.],
           [0.3, 0.0, 0., 0., 0.7],
           [0.3, 0.0, 0., 0., 0.7]]

prob[1] = [[1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.]]

rewards = np.zeros((5, 2))
rewards[0] = [0., 0.]
rewards[1] = [0., 1.]
rewards[2] = [0., 1.]
rewards[3] = [0., 1.]
rewards[4] = [0.3, 2.]

vi = mdptoolbox.mdp.ValueIteration(prob, rewards, 0.9)
vi.run()

optimal_policy = vi.policy
expected_values = vi.V

print(optimal_policy)
print(expected_values)

