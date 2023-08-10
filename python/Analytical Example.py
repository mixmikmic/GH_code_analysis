import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().magic('matplotlib notebook')

get_ipython().magic('cd ..')

def oracle(x):
    assert -3 <= x[0] <= 3, 'oracle must be queried within X'
    assert -3 <= x[1] <= 3, 'oracle must be queried within X'
    # compute function value a subgradient
    if x[0] > abs(x[1]):
        f_x = 5*(9*x[0]**2 + 16*x[1]**2)**(float(1)/float(2))
        diff_f_x = np.array([float(9*5*x[0])/np.sqrt(9*x[0]**2 + 16*x[1]**2), 
                             float(16*5*x[0])/np.sqrt(9*x[0]**2 + 16*x[1]**2)])
    else:
        f_x = 9*x[0] + 16*abs(x[1])
        if x[1] >= 0:
            diff_f_x = np.array([9, 16], dtype=float)
        else:
            diff_f_x = np.array([9, -16], dtype=float)
    return 0, -f_x, -diff_f_x  # return negation to minimize


def projection_function(x):
    # projection on the box is simply saturating the entries
    return np.array([min(max(x[0],-3),3), min(max(x[1],-3),3)])

from nsopy import SGMDoubleSimpleAveraging as DSA
from nsopy import SGMTripleAveraging as TA
from nsopy import SubgradientMethod as SG
from nsopy import UniversalPGM as UPGM
from nsopy import UniversalDGM as UDGM
from nsopy import UniversalFGM as UFGM
from nsopy import GenericDualMethodLogger

# method = DSA(oracle, projection_function, dimension=2, gamma=0.5)
# method = TA(oracle, projection_function, dimension=2, variant=2, gamma=0.5)
# method = SG(oracle, projection_function, dimension=2)
method = UPGM(oracle, projection_function, dimension=2, epsilon=10, averaging=True)
# method = UDGM(oracle, projection_function, dimension=2, epsilon=1.0)
# method = UFGM(oracle, projection_function, dimension=2, epsilon=1.0)

method_logger = GenericDualMethodLogger(method)
# start from an different initial point
x_0 = np.array([2.01,2.01])
method.lambda_hat_k = x_0

for iteration in range(100):
    method.dual_step()

box = np.linspace(-3, 3, 31)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(np.array([x_1 for x_1 in box for x_2 in box]),
                np.array([x_2 for x_1 in box for x_2 in box]),
                np.array([-oracle([x_1, x_2])[1] for x_1 in box for x_2 in box]))
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x)$')

plt.plot([x[0] for x in method_logger.lambda_k_iterates],
         [x[1] for x in method_logger.lambda_k_iterates],
         [-f_x for f_x in method_logger.d_k_iterates], 'r.-')

