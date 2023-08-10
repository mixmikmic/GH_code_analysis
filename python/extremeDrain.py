from pandas import *
import pandas as pd
from collections import OrderedDict

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 10

inh = pd.read_pickle('idf.pickle')

ag = inh.plot(color=['orange', 'green', 'purple', 'magenta', 'blue','red'])

ag.set_yscale('log')
ag.set_xscale('log')

plt.xlim([0.1,100])
plt.ylim([1, 1000])
plt.grid(True,which="both", ls="-")

plt.xlabel("Duration (Hour)", fontsize=20, fontweight = 'bold')
plt.ylabel("Rainfall Intensity (mm/hr)", fontsize=20,fontweight = 'bold')

T = 10.0
D = 0.1
a = 29.725577
b = 0.156819
c = 0.838008
k = 0.493790

Intensity_mm = a*T**k/(D + b)**c
Intensity_in = Intensity_mm/25.4
Intensity_in

A = 80000.0
npipe = A/10000.0 
npipe

Qu = 0.0104 * Intensity_in
Qu

Qn = Qu * A
Qn

Qd = Qn /npipe

d_lv = 0.6*Qd**0.377
print([Qd,d_lv])

S   = 0.5
d_h = 0.53*S**-0.188*Qd**0.377
d_h

S   = 0.125
d_h = 0.53*S**-0.188*Qd**0.377
d_h



