get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

#!mkdir /workspace/GeolProc/OTR

Yaxis = np.arange(-500.,7750.,150)

Xaxis = np.arange(0.,1250.,150)

print len(Xaxis),len(Yaxis)
print Xaxis
print Yaxis

Z1axis = np.zeros(len(Yaxis))
for i in range(len(Yaxis)):
    if Yaxis[i] <= 0:
        Z1axis[i] = -39
    elif Yaxis[i] > 0 and Yaxis[i] <= 1200:
        z = np.exp(0.0027*Yaxis[i])-40.
        Z1axis[i] = z
    elif Yaxis[i] > 1200 and Yaxis[i] <= 1600:
        Z1axis[i] = Z1axis[i-1]
    elif Yaxis[i] > 1600 and Yaxis[i] <= 3500 :
        z = (np.exp(-(Yaxis[i]-1600)/1000)-1.)*10.-17
        if z >= -23:
            Z1axis[i] = z
        else:
            Z1axis[i] = -23.
    elif Yaxis[i] > 3500 and Yaxis[i] <= 5400 :
        z = (np.exp(-(5400-Yaxis[i])/1000)-1.)*10.-15
        if z >= -23:
            Z1axis[i] = z
        else:
            Z1axis[i] = -23.
    elif Yaxis[i] > 5400 and Yaxis[i] <= 6000 :
        Z1axis[i] = Z1axis[i-1]
    elif Yaxis[i] > 6000 and Yaxis[i] <= 7200 :
        z = np.exp(0.0027*(7200-Yaxis[i]))-40.
        Z1axis[i] = z
        id = i
    else:
        Z1axis[i] = -39
Z1axis += 15

fig = plt.figure(figsize=(10,5))
ax = plt.axes(xlim=(min(Yaxis), max(Yaxis)), ylim=(min(Z1axis)-3, max(Z1axis)+10))
plt.title('Simple OTR profile', fontsize=12)
ax.set_ylabel('elevation [m]', fontsize=12)
ax.set_xlabel('lenght [m]', fontsize=12)
ax.plot(Yaxis,Z1axis,'-', lw=3,color=[139./255.,131./255.,120./255.])
ax.fill_between(Yaxis, Z1axis, min(Z1axis)-3, where=Z1axis >= min(Z1axis)-10, facecolor=[1.0,0.9,0.6], interpolate=True)
sea = 0
ax.fill_between(Yaxis, Z1axis, sea, where= sea > Z1axis, facecolor=[0.7,0.9,1.0], interpolate=True)


plt.setp(ax.get_xticklabels(), fontsize=10)
plt.setp(ax.get_yticklabels(), fontsize=10)
plt.show()

# Initialise numpy arrays
x = np.zeros((len(Yaxis),len(Xaxis)))
y = np.zeros((len(Yaxis),len(Xaxis)))
z = np.zeros((len(Yaxis),len(Xaxis)))

# Define arrays values
for i in range(len(Xaxis)):
    for j in range(len(Yaxis)):
        x[j,i] = Xaxis[i]
        y[j,i] = Yaxis[j]
        z[j,i] = Z1axis[j]

topofile = 'OTR/otr'+str(len(Yaxis))+'x'+str(len(Xaxis))+'.top'
np.savetxt(topofile, z, delimiter=' ', fmt='%g')
print 'file saved in :',topofile



