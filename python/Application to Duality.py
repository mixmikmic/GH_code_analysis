import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().magic('matplotlib notebook')

get_ipython().magic('cd ..')

from tests.analytical_oracles import AnalyticalExampleInnerProblem
from methods.quasi_monotone_subgradient_methods import SGMDoubleSimpleAveraging as DSA
from methods.quasi_monotone_subgradient_methods import SGMTripleAveraging as TA
from methods.subgradient_methods import SubgradientMethod as SG
from methods.universal_gradient_methods import UniversalPGM as UPGM
from methods.universal_gradient_methods import UniversalDGM as UDGM
from methods.universal_gradient_methods import UniversalFGM as UFGM
from methods.method_loggers import GenericDualMethodLogger

inner_problem = AnalyticalExampleInnerProblem()
dual_method = UPGM(inner_problem.oracle, inner_problem.projection_function, epsilon=0.01)
method_logger = GenericDualMethodLogger(dual_method)

for iteration in range(60):
    dual_method.dual_step()

box = np.linspace(0, 3, 30)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(np.array([lmd_1 for lmd_1 in box for lmd_2 in box]),
                np.array([lmd_2 for lmd_1 in box for lmd_2 in box]),
                np.array([inner_problem.oracle(np.array([lmd_1, lmd_2]))[1] for lmd_1 in box for lmd_2 in box]))
ax.set_xlabel('$\lambda_1$')
ax.set_ylabel('$\lambda_2$')
ax.set_zlabel('$d(\lambda)$')

plt.plot([lmd[0] for lmd in method_logger.lambda_k_iterates],
         [lmd[1] for lmd in method_logger.lambda_k_iterates],
         [d_lmd for d_lmd in method_logger.d_k_iterates], 'r.-')

