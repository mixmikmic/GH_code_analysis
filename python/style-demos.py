# All imports and setups 

get_ipython().magic('run common/import_all.py')

from common.setup_notebook import *

config_ipython()
setup_matplotlib(matplotlib_file_path='styles_files/matplotlibrc.json')
set_css_style('styles_files/custom.css')

plt.plot([i for i in range(10)], [i**2 for i in range(10)], label='$x^2$')
plt.title('A matplotlib plot')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show();



