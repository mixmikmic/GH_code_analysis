get_ipython().magic('matplotlib notebook')
from CG import *

A = np.matrix([[3.0, 2.0], [2.0, 6.0]])
b = np.matrix([[2.0], [-8.0]])
c = 0.0

fig6(A, b, c)

fig7(A, b, c)

fig8(A, b, c)

axB = fig_B()

sliders_figB(axB)

