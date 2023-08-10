get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(9,10))
c = 1
for n in range(10,20):
    j = np.linspace(0,n-1,n)
    theta = (2*j+1)*np.pi/(2*n)
    x = np.cos(theta)
    y = 0*x
    plt.subplot(10,1,c)
    plt.plot(x,y,'.')
    plt.xticks([]); plt.yticks([])
    plt.ylabel(str(n))
    c += 1

