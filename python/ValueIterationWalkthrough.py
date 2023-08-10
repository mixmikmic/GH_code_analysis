from mdp_matrix import GridWorld
from value_iteration_matrix import ValueIteration, GaussSeidelValueIteration, JacobiValueIteration, PrioritizedSweepingValueIteration, GaussSeidelJacobiValueIteration
from policy_iteration import policy_iteration, modified_policy_iteration, policy_iteration_by_inversion
import matplotlib.pyplot as plt
import numpy as np
import pprint
from operator import itemgetter
from IPython.display import Image
from IPython.core.display import HTML 
from IPython.display import Latex

Image(url= "https://webdocs.cs.ualberta.ca/~sutton/book/ebook/figtmp15.png")

test_rewards = [[i, j, -1] for i in range(5) for j in range(5)]
test_rewards[2] = [0, 2, 1]
test_rewards[23] = [4,3,1]

# Instantiate the gridworld
gw = GridWorld(5, test_rewards)
pprint.pprint(np.reshape(gw.R, (5,5)))

get_ipython().magic('matplotlib inline')

vl = ValueIteration(gw)

print("First run to optimal value at epsilon 0.001....")
optimal_policy, optimal_value, _  = vl.run()

print("Now evaluating policies...\n")

print("Value Iteration")
optimal_policy, v, vs = vl.run(optimal_value=optimal_value, theta=0.01)
plt.plot(vs)

print("\nGS")

vl = GaussSeidelValueIteration(gw)

optimal_policy, v, vsgs = vl.run(optimal_value=optimal_value, theta=0.01)
plt.plot(vsgs)

print("\nJ")
vl = JacobiValueIteration(gw)
optimal_policy, v, vsj = vl.run(optimal_value=optimal_value, theta=0.01)
plt.plot(vsj)

print("\nGSJ")
vl = GaussSeidelJacobiValueIteration(gw)
optimal_policy, v, vsj = vl.run(optimal_value=optimal_value, theta=0.01)
plt.plot(vsj)

print("\nPS")
vl = PrioritizedSweepingValueIteration(gw)
optimal_policy, v, vsps = vl.run(optimal_value=optimal_value)
plt.plot(vsps)

plt.xlabel('state iterations')
plt.ylabel('||v - v*||',color='b')

plt.legend(['Value Iteration', 'Gauss-Seidel', 'Jacobi', 'Gauss-Seidel-Jacobi', 'Prioritized Sweeping Value Iteration'], loc='upper right')

plt.show()

print("Example optimal policy")
print(np.array([gw.actions[x] for x in optimal_policy.values()]).reshape((5,5)))

import numpy
import scipy.linalg

max_alpha = None
for a in range(gw.A):
    L, U = numpy.tril(gw.T[:,a,:], k=-1), numpy.triu(gw.T[:,a,:])
#     print(numpy.diag(L))

    Q = np.eye(gw.S) - .9*L
    R = .9*U
#     print(numpy.linalg.inv(Q).dot(R))
    w = numpy.linalg.eigvals(numpy.linalg.inv(Q).dot(R))
#     print(w)
#     print(V)
    alpha = max(w)
    if not max_alpha or alpha > max_alpha:
        max_alpha = alpha

print(max_alpha)
    

import numpy
import scipy.linalg

max_alpha = None
for a in range(gw.A):
    copy = gw.T[:,a,:].copy()
    np.fill_diagonal(copy, 0.)
    R, Q = .9*copy, numpy.diag(1. - .9*numpy.diag(gw.T[:,a,:]))
#     print(numpy.diag(Q))
    QRinv = numpy.linalg.inv(Q).dot(R)
    
    w = numpy.linalg.eigvals(QRinv)
    alpha = max(w)
    if not max_alpha or alpha > max_alpha:
        max_alpha = alpha


Image(filename="book_iterations.png")

get_ipython().run_cell_magic('latex', '', '\\begin{align} \nV &=PR + \\gamma PV\\\\\n\\Leftrightarrow (I-\\gamma P) V &= PR\\\\\n\\Rightarrow V &= (I-\\gamma P)^{-1} PR \n\\end{align}')

get_ipython().magic('matplotlib inline')
epsilon = 0.01
gamma = 0.9

m = 5

V_pi, pol_pi, n_iter_pi, vs_pi = policy_iteration(gw, gamma, epsilon, optimal_value)
plt.plot(vs_pi)

V_inv, pol_inv, n_iter_inv, vs_inv = policy_iteration_by_inversion(gw, gamma, optimal_value)
plt.plot(vs_inv)

V_modpi, pol_modpi, n_itermodpi, vs_modpi = modified_policy_iteration(gw, gamma, epsilon, m, optimal_value)
plt.plot(vs_modpi)

plt.xlabel('state evaluations')
plt.ylabel('||v - v*||',color='b')

plt.legend(['Policy Iteration', 'Policy Iteration by Matrix Inversion', 'Modified Policy Iteration'], loc='upper right')

plt.show()

dim = 10
test_rewards_2 = [[i, j, -1] for i in range(dim) for j in range(dim)]
test_rewards_2[0] = [0, 0, 1]
test_rewards_2[99] = [9, 9 ,1]

# Instantiate the gridworld
gw2 = GridWorld(dim, test_rewards)

get_ipython().magic('matplotlib inline')

import time

vl = ValueIteration(gw2)

print("Value Iteration")

optimal_policy, optimal_value, _  = vl.run()
time1 = time.time()
optimal_policy, v, vs = vl.run(optimal_value=optimal_value, theta=0.01)
time2 = time.time()
print 'took %0.3f ms' % ((time2-time1)*1000.0)
plt.plot(vs)

vl = GaussSeidelValueIteration(gw2)

print("\nGS Value Iteration")

time1 = time.time()
optimal_policy, v, vsgs = vl.run(optimal_value=optimal_value, theta=0.01)
time2 = time.time()
print 'took %0.3f ms' % ((time2-time1)*1000.0)
plt.plot(vsgs)

print("\nJacobi Value Iteration")

vl = JacobiValueIteration(gw2)
time1 = time.time()
optimal_policy, v, vsj = vl.run(optimal_value=optimal_value, theta=0.01)
time2 = time.time()
print 'took %0.3f ms' % ((time2-time1)*1000.0)
plt.plot(vsj)

print("\nGSJ Value Iteration")

vl = GaussSeidelJacobiValueIteration(gw2)
time1 = time.time()
optimal_policy, v, vsj = vl.run(optimal_value=optimal_value, theta=0.01)
time2 = time.time()
print 'took %0.3f ms' % ((time2-time1)*1000.0)
plt.plot(vsj)

print("\nPrioritized Sweeping Iteration")

vl = PrioritizedSweepingValueIteration(gw2)
time1 = time.time()
optimal_policy, v, vsps = vl.run(optimal_value=optimal_value, max_iterations=100000)
time2 = time.time()
print 'took %0.3f ms' % ((time2-time1)*1000.0)
plt.plot(vsps)

print("\nPolicy Iteration")

time1 = time.time()
V_pi, pol_pi, n_iter_pi, vs_pi = policy_iteration(gw2, gamma, epsilon, optimal_value)
time2 = time.time()
print 'took %0.3f ms' % ((time2-time1)*1000.0)
plt.plot(vs_pi)

print("\nModified Policy Iteration")

time1 = time.time()
V_modpi, pol_modpi, n_itermodpi, vs_modpi = modified_policy_iteration(gw2, gamma, epsilon, m, optimal_value)
time2 = time.time()
print 'took %0.3f ms' % ((time2-time1)*1000.0)
plt.plot(vs_modpi)

plt.xlabel('State Iterations')
plt.ylabel('||v - v*||',color='b')

plt.legend(['Value Iteration', 'Gauss-Seidel', 'Jacobi',
            'Gauss-Seidel-Jacobi', 'Prioritized Sweeping Value Iteration',
            'Policy Iteration',
            'Modified Policy Iteration'], loc='upper right')

plt.show()



