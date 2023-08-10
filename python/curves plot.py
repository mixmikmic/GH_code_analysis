import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os
import collections
import matplotlib as mp
mp.style.use('classic')
#mp.rcParams.update(mp.rcParamsDefault)

# Read simulation files
def datafileread(measurename,skipfirstrows,delim='\t'):
    # Reading Datafiles
    path = measurename
    data = np.genfromtxt(path,
                        skip_header=skipfirstrows,
                        delimiter=delim,
                        dtype=(float,float),
                        unpack=True)
    return data

# 
time, v0, v1, v2, v3, v4 = datafileread('tline_comparison.csv',1,delim=" ")
time *= 1e9

#
f, ax1 = plt.subplots(1,1,figsize=(10,4))
xlims = [-20, 280]
ax1.plot(time, v0, label="behavioral")
ax1.plot(time, v1, label="distributed N=10")
ax1.plot(time, v2, label="distributed N=100")
ax1.plot(time, v3, label="distributed N=1000")
#ax1.plot(time, v4, label="distributed N=10000")# Problem with this simulation
#ax1.set_xlim(xlims)
#ax1.set_ylim([1, 13])
#ax1.set_title('TLP output characterization')
#ax1.get_xaxis().set_ticklabels([])
ax1.set_ylabel('Voltage (V)')
ax1.set_xlabel('Time (ns)')

#
plt.tight_layout()
plt.legend(loc="best")
plt.tight_layout()
f.subplots_adjust(hspace=0)
plt.savefig("../../src/2/figures/tline_comparison.png", pad_inches=0.3)
plt.show()

