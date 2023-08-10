import wfdb
import numpy as np
import matplotlib.pyplot as plt
import sys

folder = "p05/p050140"
waveform = "p050140-2188-07-26-05-51"
sig, fields = wfdb.srdsamp(waveform,pbdir='mimic3wdb/matched/'+folder, sampto=150000)

print("signame: " + str(fields['signame']))
print("units: " + str(fields['units']))
print("fs: " + str(fields['fs']))
print("comments: " + str(fields['comments']))
print("fields: " + str(fields))

signalII = None
try:
    signalII = fields['signame'].index("II")
except ValueError:
    print("List does not contain value")
if(signalII!=None):
    print("List contain value")

array = sig[:, signalII]
array = np.nan_to_num(array)

fs = fields['fs']
fs

npArray = np.array(array)
npArray

from oct2py import octave
octave.addpath('/home/scidb/HeartRatePatterns/Matlab/')
octave.eval('pkg load signal')

get_ipython().run_line_magic('load_ext', 'oct2py.ipython')

get_ipython().run_cell_magic('octave', '-i array,fs -o R_i,R_amp,S_i,S_amp,T_i,T_amp,Q_i,Q_amp,heart_rate,buffer_plot', '[R_i,R_amp,S_i,S_amp,T_i,T_amp,Q_i,Q_amp,heart_rate,buffer_plot] = peakdetect(array,fs,false); ')

print("R_i",R_i)
print("S_i",S_i)
print("R_i",T_i)
print("Q_i",Q_i)
print(heart_rate)
print(buffer_plot)

To= 5
subarray = buffer_plot[0][Q_i[0][0]:T_i[0][To]]
print(len(subarray))
subQ_i = Q_i[0][:To]
subQ_amp = Q_amp[0][:To]
subR_i = R_i[0][:To]
subR_amp = R_amp[0][:To]
subS_i = S_i[0][:To]
subS_amp = S_amp[0][:To]
subT_i = T_i[0][:To]
subT_amp = T_amp[0][:To]

fig_size = [12,9]
plt.rcParams["figure.figsize"] = fig_size

plt.plot(np.arange(Q_i[0][0],T_i[0][To]),np.array(subarray),c='lightgreen')
plt.scatter(subQ_i, subQ_amp,c='yellow')
plt.scatter(subR_i, subR_amp,c='blue')
plt.scatter(subS_i, subS_amp,c='red')
plt.scatter(subT_i, subT_amp,c='black')
plt.show()

qt = Q_i[0][:209]-T_i
qt

ts = T_i-S_i[0][:209]
ts

sr = S_i-R_i
sr



