get_ipython().magic('reload_ext pytriqs.magic')

get_ipython().run_cell_magic('triqs', '            ', '#include <triqs/gfs.hpp>\nusing namespace triqs;\nusing namespace triqs::gfs;\n        \ngf<imfreq> demo(double beta) {\n\n int n_freq = 1000;\n\n clef::placeholder<0> iw_;\n\n // Construction of a 1x1 matrix-valued fermionic gf on Matsubara frequencies.\n auto g_iw = gf<imfreq>{{beta, Fermion, n_freq}, {1, 1}};\n\n // Automatic placeholder evaluation\n g_iw(iw_) << 1 / (iw_ + 2);\n\n // Writing the Green\'s functions into an HDF5 file.\n auto f = h5::file("gf.h5", \'w\');\n h5_write(f, "g_iw", g_iw);\n\n return g_iw;\n}')

get_ipython().system('rm gf.h5')
g = demo(10.0)

get_ipython().system('h5ls -r gf.h5')

from pytriqs.gf.local import *
from pytriqs.archive import *
from pytriqs.plot.mpl_interface import oplot, oplotr
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
#with HDFArchive("gf.h5",'r') as A:  # Open file
#  oplot(A['g_iw'], '-o')
oplot(demo(30), '-p')
plt.xlim(0,10)
#plt.ylim(-0.5,0)



