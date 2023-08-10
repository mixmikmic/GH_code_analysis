get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
from numpy import exp, arctan, linspace, abs
import matplotlib.pyplot as plt

f = lambda x: exp(x) - 3.0/2.0 - arctan(x)

x = linspace(0.0, 1.0, 100)
plt.plot(x,f(x))
plt.grid(True)

M = 50 # maximum number of iterations
eps = 1.0e-6 # tolerance for stopping
x0, x1 = 0.5, 0.6

for i in range(M):
    f0, f1 = f(x0), f(x1)
    x2 = x1 - f1*(x1 - x0)/(f1 - f0)
    if abs(x2-x1) < abs(x2)*eps:
        break
    print i, x0, x1, f0, f1
    x0, x1 = x1, x2
    
print "Number of iterations = ", i
print "Final solution = ", x2, f(x2)

