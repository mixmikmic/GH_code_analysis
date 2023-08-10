# All imports and setups 

get_ipython().magic('run ../common/import_all.py')

from common.setup_notebook import *

config_ipython()
setup_matplotlib()
set_css_style()

# Plotting a power function 

x = np.array([i for i in np.arange(0.1, 1.1, 0.01)])
y = np.array([item**-0.3 for item in x])

plt.plot(x, y, label='$y = x^{-0.3}$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.xticks([i for i in np.arange(0, 1.1, 0.1)])
plt.title("A power law")
plt.legend()
plt.show();



