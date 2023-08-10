get_ipython().run_line_magic('load_ext', 'base16_mplrc')
get_ipython().run_line_magic('base16_mplrc', 'dark bespin')

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

def lorenz(in_, t, sigma, b, r):
    """Evaluates the RHS of the 3 
    Lorenz attractor differential equations.

    in_ : initial vector of [x_0, y_0, z_0]
    t : time vector (not used, but present for odeint() call)
    sigma : numerical parameter 1
    b :     numerical parameter 2
    r :     numerical parameter 3
    """
    x = in_[0]
    y = in_[1]
    z = in_[2]
    return [sigma*(y-x),
            r*x - y - x*z,
            x*y - b*z]

def get_lorenz_solution(in_0, tmax, nt, args_tuple):
    t = np.linspace(0, tmax, nt)
    soln = odeint(lorenz, in_0, t, args=args_tuple).T
    return t, soln

in_0 = [5.0, 5.0, 5.0]
t_max = 20
t_steps = 20000
t, [solx, soly, solz] = get_lorenz_solution(in_0, t_max, t_steps, 
                                            (10.0, 8/3, 28))

fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.plot(t, solx, label='x')
ax.plot(t, soly, label='y')
ax.plot(t, solz, label='z')
ax.legend()
plt.show()



