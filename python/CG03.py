get_ipython().magic('matplotlib notebook')
from CG import *

A = np.matrix([[3.0, 2.0], [2.0, 6.0]])
b = np.matrix([[2.0], [-8.0]])
c = 0.0

plotAb2D(A, b)

plotAbc3D(A, b, c)

plotcontours(A, b, c)

vectorfield(A, b, c)

fig5()

hdls = fig_A()

sliders_figA(hdls);

