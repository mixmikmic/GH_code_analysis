import matplotlib.pyplot as plt
import numpy as np

xx = np.linspace(-0.01,0.01,1000)
def f(x):
    return (x**4)*(np.cos(1/x)+2)

get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(xx,f(xx),linewidth=3)
plt.show()



