import FGS
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

datadir=''

tab=FGS.table()  #Use the 'table' function to grab the FGS data
print 'Available tags: ',list(tab.keys())

plt.hist(tab.KEPMAG[tab.mission=='kepler'],30
             ,alpha=0.5,label='kepler')
plt.hist(tab.KEPMAG[tab.mission=='k2'],30
             ,alpha=0.5,label='k2')
plt.legend()
plt.title('Kepler Magnitude Distribution')
plt.ylabel('Frequency')
plt.xlabel('KepMag')

#Get the full Kepler FGS data from MAST, with every quarter.
#(This will take several minutes to download)
#FGS.get_data(datadir,mission='kepler')


#Get the first campaign of K2 FGS data from MAST
FGS.get_data(datadir,mission='k2',quarters=1)

#Create a lightcurve for Kepler ID 7394260 using quarters 3,4 and 5
ID=7394260
time,flux,column,row=FGS.gen_lc(datadir,ID=ID,quarters=[3,4,5])
plt.figure()
plt.scatter(time,flux,s=0.1)
plt.xlabel('Time (days)')
plt.ylabel('Normalised Flux')
plt.title('Kepler ID '+str(ID))

#Create a lightcurve for K2 star 207187836
ID=207187836
time,flux,column,row=FGS.gen_lc(datadir,ID=ID)
plt.figure()
plt.scatter(time,flux,s=0.1)
plt.xlabel('Time (days)')
plt.ylabel('Normalised Flux')
plt.title('Kepler ID '+str(ID))





