import numpy as np
M = np.array([[4,5,1], [0, 11, 8]])
M

import numpy as np
M = np.array([[4,5,1], [0, 11, 8]])
N = np.array([[7],[0],[1]])
M.dot(N)

import numpy as np
M = np.array([[4,5,1], [0, 11, 8]])
M.T

import numpy as np
np.eye(3)

import numpy as np
M = np.array([[4,6], [1,7]])
M_inv = np.linalg.inv(M)
print(M_inv)
M.dot(M_inv)

