import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os
import collections
import matplotlib as mp
mp.style.use('classic')

# Read simulation files
def datafileread(measurename,skipfirstrows, delim='\t'):
    # Reading Datafiles
    path = measurename
    data = np.genfromtxt(path,
                        skip_header=skipfirstrows,
                        delimiter=delim,
                        dtype=(float,float),
                        unpack=True)
    return data

#
time_bm, v_bm, i_bm = datafileread('vi_model_-50.csv',1, delim=' ')
time_br, v_br, i_br = datafileread('vi_total_-50.csv',1, delim=' ')

#
time_gm, v_gm = datafileread('vmodel_good.csv',1)
time_gr, v_gr = datafileread('vref.csv',1)

#
time_bm *= 1e6
time_br *= 1e6
time_gm *= 1e6
time_gr *= 1e6

# 
i_bm *= -1 # wrong pin in Cadence for probing current

#
f, ax1 = plt.subplots(1,1,figsize=(10,4))
xlims = [0, 12.5]
ax1.plot(time_br, v_br, label="complete schematic")
ax1.plot(time_bm, v_bm, label="I(V) output model")
#ax1.set_xlim(xlims)
#ax1.set_ylim([0, 13])
ax1.set_title('$V_{2p5}$')
#ax1.get_xaxis().set_ticklabels([])
ax1.set_ylabel('Voltage (V)')
ax1.set_xlabel('Time (μs)')

#ax2.plot(time_br, i_br, label="complete schematic")
#ax2.plot(time_bm, i_bm, label="model")
#ax2.set_xlim(xlims)
#ax2.set_ylim([-0.01, 0.05])
#ax2.set_ylabel('current (A)')
#ax2.set_xlabel('Time (us)')

#
plt.tight_layout()
plt.legend(loc="best")
plt.tight_layout()
#f.subplots_adjust(hspace=0.2)
plt.savefig("../../src/4/figures/comparison_model_total_output_bad_m10V.png", pad_inches=0.3)
plt.show()

#
f, ax1 = plt.subplots(1,1,figsize=(8,4))
xlims = [0, 12.5]
ax1.plot(time_br, v_br, label="Schéma complet")
ax1.plot(time_bm, v_bm, label="Modèle")
#ax1.set_xlim(xlims)
#ax1.set_ylim([0, 13])
ax1.set_title('$V_{2p5}$')
#ax1.get_xaxis().set_ticklabels([])
ax1.set_ylabel('Voltage (V)')
ax1.set_xlabel('Time (μs)')

#ax2.plot(time_br, i_br, label="complete schematic")
#ax2.plot(time_bm, i_bm, label="model")
#ax2.set_xlim(xlims)
#ax2.set_ylim([-0.01, 0.05])
#ax2.set_ylabel('current (A)')
#ax2.set_xlabel('Time (us)')

#
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("./comparison_model_total_output_bad_m10V.png", pad_inches=0.3)
plt.show()

#
f, ax1 = plt.subplots(1,1,figsize=(10,4))
xlims = [0, 12.5]
ax1.plot(time_br, i_br, label="complete schematic")
ax1.plot(time_bm, i_bm, label="I(V) output model")
#ax1.set_xlim(xlims)
ax1.set_ylim([-0.01, 0.02])
#ax1.set_title('TLP output characterization')
#ax1.get_xaxis().set_ticklabels([])
ax1.set_ylabel('Current (A)')
ax1.set_xlabel('Time (us)')

#ax2.plot(time_br, i_br, label="complete schematic")
#ax2.plot(time_bm, i_bm, label="model")
#ax2.set_xlim(xlims)
#ax2.set_ylim([-0.01, 0.05])
#ax2.set_ylabel('current (A)')
#ax2.set_xlabel('Time (us)')

#
plt.tight_layout()
plt.legend(loc="best")
plt.tight_layout()
f.subplots_adjust(hspace=0.2)
#plt.savefig("../../src/4/figures/comparison_model_total_output_bad_m10V.png", pad_inches=0.3)
plt.show()

#
f, ax1 = plt.subplots(1,1,figsize=(10,4))
xlims = [0, 12.5]
ax1.plot(time_gr, v_gr, label="complete schematic")
ax1.plot(time_gm, v_gm, label="model")
#ax1.set_xlim(xlims)
#ax1.set_ylim([0, 13])
#ax1.set_title('TLP output characterization')
#ax1.get_xaxis().set_ticklabels([])
ax1.set_ylabel('Voltage (V)')
ax1.set_xlabel('Time (us)')

#ax2.plot(time_total_m10, i_total_m10, label="complete schematic")
#ax2.plot(time_model_m10, i_model_m10, label="model")
#ax2.set_xlim(xlims)
#ax2.set_ylim([-0.01, 0.01])
#ax2.set_ylabel('current (A)')
#ax2.set_xlabel('Time (us)')

#
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("../../src/4/figures/comparison_model_total_output_good_m10V.png", pad_inches=0.3)
plt.show()

#
f, ax1 = plt.subplots(1,1,figsize=(10,4))
xlims = [0, 12.5]
ax1.plot(time_gr, v_gr, label="Schéma complet")
ax1.plot(time_gm, v_gm, label="Modèle")
#ax1.set_xlim(xlims)
#ax1.set_ylim([0, 13])
#ax1.set_title('TLP output characterization')
#ax1.get_xaxis().set_ticklabels([])
ax1.set_ylabel('Voltage (V)')
ax1.set_xlabel('Time (us)')

#ax2.plot(time_total_m10, i_total_m10, label="complete schematic")
#ax2.plot(time_model_m10, i_model_m10, label="model")
#ax2.set_xlim(xlims)
#ax2.set_ylim([-0.01, 0.01])
#ax2.set_ylabel('current (A)')
#ax2.set_xlabel('Time (us)')

#
plt.tight_layout()
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("./comparison_model_total_output_good_m10V.png", pad_inches=0.3)
plt.show()

