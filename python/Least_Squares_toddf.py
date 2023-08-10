import pandas as pd
from sqlalchemy import create_engine

eng = create_engine('sqlite:///new_parkinglot.db')

df = pd.read_sql_table('training', eng)

print(df.head())

import matplotlib.pyplot as plt
import numpy as np
import math

Dist1 = []
Dist2 = []
Dist3 = []
Dist4 = []

locx = df['LOCX']
locy = df['LOCY']

for i in range(0, 600):
    Dist1.append(math.sqrt(locx[i]**2 + locy[i]**2))
    Dist2.append(math.sqrt(locx[i]**2 + (-10 - locy[i])**2))
    Dist3.append(math.sqrt((10 - locx[i])**2 + (-10 - locy[i])**2))
    Dist4.append(math.sqrt((10 - locx[i])**2 + locy[i]**2))
    

logDist1 = np.log10(Dist1)
logDist2 = np.log10(Dist2) 
logDist3 = np.log10(Dist3) 
logDist4 = np.log10(Dist4) 

ones = []
    
for i in range(0, 600):
    ones.append(1)

nuc1_mat = np.column_stack((ones, logDist1))
nuc2_mat = np.column_stack((ones, logDist2))
nuc3_mat = np.column_stack((ones, logDist3))
nuc4_mat = np.column_stack((ones, logDist4))

# Solve for a & b using the least squares function in numpy

b1, a1 = np.linalg.lstsq(nuc1_mat, df['NUC1'])[0]
b2, a2 = np.linalg.lstsq(nuc2_mat, df['NUC2'])[0]
b3, a3 = np.linalg.lstsq(nuc3_mat, df['NUC3'])[0]
b4, a4 = np.linalg.lstsq(nuc4_mat, df['NUC4'])[0]

print(a1, b1)
print(a2, b2)
print(a3, b3)
print(a4, b4)

y1 = []
y2 = []
y3 = []
y4 = []

for i in range (0, 600):
    y1.append(a1*logDist1[i] + b1)
    y2.append(a2*logDist2[i] + b2)
    y3.append(a3*logDist3[i] + b3)
    y4.append(a4*logDist4[i] + b4)

lin1 = np.transpose(y1)
lin2 = np.transpose(y2)
lin3 = np.transpose(y3)
lin4 = np.transpose(y4)

# Plot graphs

#plt.figure(1)
plt.subplot(221)
plt.plot(logDist1, df['NUC1'], 'o')
plt.plot(logDist1, lin1)
plt.title("NUC1")

#plt.figure(2)
plt.subplot(222)
plt.plot(logDist2, df['NUC2'], 'o')
plt.plot(logDist2, lin2)
plt.title("NUC2")

#plt.figure(3)
plt.subplot(223)
plt.plot(logDist3, df['NUC3'], 'o')
plt.plot(logDist3, lin3)
plt.title("NUC3")

#plt.figure(4)
plt.subplot(224)
plt.plot(logDist4, df['NUC4'], 'o')
plt.plot(logDist4, lin4)
plt.title("NUC4")

plt.subplots_adjust(top=1, bottom=0.25, left=0.10, right=0.95, hspace=0.5, wspace=0.35)

plt.show()

