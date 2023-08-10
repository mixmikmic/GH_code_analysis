# Set up matrix K and Y
import numpy as np
from scipy.linalg import solve
from numpy.linalg import inv
import time
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

n = 1000
K = np.random.randn(n, n)
K = K.dot(K.T)
Y = np.random.randn(n, n)

def generate(n):
    K = np.random.randn(n, n)
    K = K.dot(K.T)
    Y = np.random.randn(n, n)
    return K, Y

def solve_by_inv(K, Y):
    # Solve by matrix inversion
    t = time.time()
    X = inv(K).dot(Y)
    t = time.time() - t
    err = abs(K.dot(X) - Y).mean()
    return err, t

def solve_by_sol(K, Y):
    # Solve by matrix inversion
    t = time.time()
    X = solve(K, Y, sym_pos=True)
    t = time.time() - t
    err = abs(K.dot(X) - Y).mean()
    return err, t    

# Solve by matrix inversion
err, t = solve_by_inv(K, Y)
print "Solution check:", err
print "Computation time:", t

# Solve by solving
err, t = solve_by_sol(K, Y)
print "Solution check:", err
print "Computation time:", t

ns = np.power(10, np.linspace(1, 3.5, num=10)).astype(int)
trials = 3
inv_err = np.zeros(trials*len(ns))
sol_err = np.zeros(trials*len(ns))
inv_t = np.zeros(trials*len(ns))
sol_t = np.zeros(trials*len(ns))
x = np.zeros(trials*len(ns))

i = 0
for n in ns:
    print n
    for _ in xrange(trials):
        K, Y = generate(n)
        err1, t1 = solve_by_inv(K, Y)
        err2, t2 = solve_by_sol(K, Y)
        inv_err[i] = err1
        sol_err[i] = err2
        inv_t[i] = t1
        sol_t[i] = t2
        x[i] = n
        i += 1
        print "*"*80
        print "Error:", err1, "\t", err2
        print "Time:", t1, "\t", t2

y1 = np.array([inv_err[x == n].mean() for n in ns])
y2 = np.array([sol_err[x == n].mean() for n in ns])
plt.axis((x.min(), x.max(), 0, max(y1.max(),y2.max())))
plt.plot(ns, y1, color='r')
plt.plot(ns, y2, color='g')
plt.title("Error rate")
plt.show()

y1 = np.array([inv_t[x == n].mean() for n in ns])
y2 = np.array([sol_t[x == n].mean() for n in ns])
plt.axis((x.min(), x.max(), 0, max(y1.max(),y2.max())))
plt.plot(ns, y1, color='r')
plt.plot(ns, y2, color='g')
plt.title("Compute time")
plt.show()

