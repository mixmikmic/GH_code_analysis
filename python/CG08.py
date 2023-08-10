get_ipython().magic('matplotlib notebook')
from CG import *

A = np.matrix([[3.0, 2.0], [2.0, 6.0]])
b = np.matrix([[2.0], [-8.0]])
c = 0.0

fig29()

fig30(A, b, c)

axC = fig_C()

sliders_figC(axC)

