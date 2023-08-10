get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvalsh
from collections import namedtuple

import TB

TB.band(TB.Si)

TB.band(TB.GaAs)

TB.band(TB.Ge)

def SiGe_band(x=0.2):
    Si_data = TB.bandpts(TB.Si)
    Ge_data = TB.bandpts(TB.Ge)
    data = (1-x)*Si_data + x*Ge_data
    TB.bandplt("SiGe, %%Ge=%.2f" % x,data)
    return

SiGe_band(0)

SiGe_band(0.1)

SiGe_band(0.25)

SiGe_band(0.37)

Ge_CB = TB.bandpts(TB.Ge)[:,4]
Si_CB = TB.bandpts(TB.Si)[:,4]

nk = len(Si_CB)
n = (nk-2)//3

plt.plot(Si_CB)
plt.plot(Ge_CB)
TB.band_labels(n)
plt.axis(xmax=3*n+1)

plt.plot(Si_CB,label='Si')
plt.plot(Ge_CB,label='Ge')
plt.plot(0.9*Si_CB + 0.1*Ge_CB,label='Si_0.9 Ge_0.1')
TB.band_labels(n)
plt.axis(xmax=3*n+1)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

min_Si = min(Si_CB)
min_Ge = min(Ge_CB)
print min_Si, min_Ge
# min_Si - min_Ge = 0.12
Si_CB_shifted = Si_CB - min_Si + min_Ge + 0.12



