import sys
#change to match where the PmagPy folder is on your computer
sys.path.insert(0, '/Users/Laurentia/PmagPy')
import pmag,pmagplotlib,ipmag # import PmagPy functions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')

Rose_Hill_Plat = -19.1 
Rose_Hill_Plong = 308.3
Rose_Hill = (Rose_Hill_Plong, Rose_Hill_Plat)

Slate_Lat = 48.6
Slate_Long = -87.0
Slate = (Slate_Long, Slate_Lat)

plat = 90 - pmag.angle(Rose_Hill, Slate)
print plat[0]

VGPs = ipmag.tk03(n=100,lat=0) #set at 100,000 this takes a long time to run
VGP_dataframe = pd.DataFrame(VGPs,columns=['dec_tc','inc_tc','int'])
VGP_dataframe['site_lat'] = pd.Series(np.random.uniform(0,0,size=len(VGPs)))
VGP_dataframe['site_lon'] = pd.Series(np.random.uniform(0,0,size=len(VGPs)))
  
ipmag.vgp_calc(VGP_dataframe)    

VGP_dataframe.head()

greater_10 = []
greater_20 = []
greater_30 = []
greater_40 = []
greater_50 = []
angles = []
for n in range(len(VGP_dataframe)):
    true_north = (0,90)
    vgp = (VGP_dataframe['vgp_lon'][n],VGP_dataframe['vgp_lat'][n])
    angle = pmag.angle(true_north,vgp)
    angles.append(angle[0])
    if angle > 10:
        greater_10.append(angle[0])
    if angle > 20:
        greater_20.append(angle[0])
    if angle > 30:
        greater_30.append(angle[0])
    if angle > 40:
        greater_40.append(angle[0])
    if angle > 50:
        greater_50.append(angle[0])
    
plt.hist(angles, bins=50)
plt.xlim(0,90)
plt.vlines(50.6,0,5000,linestyles='dotted')
plt.xlabel('angle between TK03.GAD VGP and geographic north')
plt.savefig('SV_histogram.pdf')
plt.show()

print "Percent of VGPS with angular difference from mean greater than 10º"
print float(len(greater_10))/float(len(angles))*100.0
print "Percent of VGPS with angular difference from mean greater than 20º"
print float(len(greater_20))/float(len(angles))*100.0
print "Percent of VGPS with angular difference from mean greater than 30º"
print float(len(greater_30))/float(len(angles))*100.0
print "Percent of VGPS with angular difference from mean greater than 40º"
print float(len(greater_40))/float(len(angles))*100.0
print "Percent of VGPS with angular difference from mean greater than 50º"
print float(len(greater_50))/float(len(angles))*100.0

gauss = pmag.mktk03(8,1,0,0)
pmag.getvec(gauss,0,0)



