import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
plotly.offline.init_notebook_mode()
import plotly.plotly as py
import plotly.graph_objs as go

with open("A00051826_vis_learn Events.txt") as eyetracking:
        lines = [line.split() for line in eyetracking]

print len(lines)
print lines

uec = 0
bsfc = 0 
for i in range(len(lines)):
    if (lines[i] <> []):
        if len(lines[i]) == 4 and (lines[i][3]) == "Fixations:":
            FLabels = lines[i+1]
        if len(lines[i]) == 4 and (lines[i][3]) == "Saccades:":
            SLabels = lines[i+1]
        if len(lines[i]) == 4 and (lines[i][3]) == "Blinks:":
            BLabels = lines[i+1]
        if (lines[i][0]) == "UserEvent":
            if uec == 1:
                uec += 1
                UE = lines[i]
                uel = len(lines[i])
            if uec == 0:
                uel = 0
                uec += 1
            if uel == len(lines[i]) and (lines[i][2] <> '2'):
                UE = np.vstack((UE, lines[i]))
        if (lines[i][0] == "Saccade" or lines[i][0] == "Fixation" or lines[i][0] == "Blink") and (bsfc == 0):
            bsfc = bsfc+1
            BSF = lines[i:]
print len(BSF)

SL = []
SR = []
BL = []
BR = []
FL = []
FR = []
slc = 0
blc = 0
flc = 0
src = 0
brc = 0
frc = 0

for i in range(len(BSF)):
    if (BSF[i][0] == "Saccade") and (BSF[i][1] == "L"):
        if (slc == 0):
            slc += 1
            SL = np.full(len(BSF[i]), 0, dtype = int)
        SL = np.vstack((SL,BSF[i]))
    if (BSF[i][0] == "Blink") and (BSF[i][1] == "L"):
        if (blc == 0):
            blc += 1
            BL = np.full(len(BSF[i]), 0, dtype = int)
        BL = np.vstack((BL,BSF[i]))
    if (BSF[i][0] == "Fixation") and (BSF[i][1] == "L"):
        if (flc == 0):
            flc += 1
            FL = np.full(len(BSF[i]), 0, dtype = int)
        FL = np.vstack((FL,BSF[i]))
    if (BSF[i][0] == "Saccade") and (BSF[i][1] == "R"):
        if (src == 0):
            src += 1
            SR = np.full(len(BSF[i]), 0, dtype = int)
        SR = np.vstack((SR,BSF[i]))
    if (BSF[i][0] == "Blink") and (BSF[i][1] == "R"):
        if (brc == 0):
            brc += 1
            BR = np.full(len(BSF[i]), 0, dtype = int)
        BR = np.vstack((BR,BSF[i]))
    if (BSF[i][0] == "Fixation") and (BSF[i][1] == "R"):
        if (frc == 0):
            frc += 1
            FR = np.full(len(BSF[i]), 0, dtype = int)
        FR = np.vstack((FR,BSF[i]))
        
SL = SL[1:,:]
BL = BL[1:,:]
FL = FL[1:,:]
SR = SR[1:,:]
BR = BR[1:,:]
FR = FR[1:,:]
print SLabels
print BLabels
print FLabels
print SL

def distance(saccade):
    tp = len(saccade[:,0])
    xdist = np.full(tp,0, dtype=float)
    ydist = np.full(tp,0, dtype=float)
    for i in range(tp):
        xdist[i] = float(saccade[i][8]) - float(saccade[i][6])
        ydist[i] = float(saccade[i][9]) - float(saccade[i][7])
    return saccade, xdist, ydist
distance(SL)

print SL[0][0]
print SL[0][1]
print SL[0][2]
print SL[0][3]
print SL[0][4]
print SL[0][5]
print SL[0][6]
print SL[0][7]
plt.plot(SL[:,4], SL[:,7])
plt.plot(FL[:,4], FL[:,7])
plt.xlabel('Starting Times')
plt.ylabel('X Location of the Left Eye')
plt.show()

plt.plot(SR[:,4], SR[:,7])
plt.plot(FR[:,4], FR[:,7])
plt.xlabel('Starting Times')
plt.ylabel('X Location of the Right Eye')
plt.show()



