get_ipython().magic('run ../common/import_all.py')

from common.setup_notebook import set_css_style, setup_matplotlib, config_ipython
config_ipython()
setup_matplotlib()
set_css_style()

p = np.arange(0,1.05,0.05)
I = 1./(1 + 2*p**2 -2*p)

plt.plot(p, I)
plt.xlabel('$p$')
plt.ylabel('$I$')
plt.show();



