get_ipython().magic('matplotlib inline')
from __future__ import division
import os
import nibabel as nib
import numpy as np
from neuropower import peakdistribution
import scipy.integrate as integrate
import pandas as pd
import matplotlib.pyplot as plt
import palettable.colorbrewer as cb

if not 'FSLDIR' in os.environ.keys():
    raise Exception('This notebook requires that FSL is installed and the FSLDIR environment variable is set')


# From smoothness + mask to ReselCount

FWHM = 3
ReselSize = FWHM**3
MNI_mask = nib.load(os.path.join(os.getenv('FSLDIR'),'data/standard/MNI152_T1_2mm_brain_mask.nii.gz')).get_data()
Volume = np.sum(MNI_mask)
ReselCount = Volume/ReselSize

print("ReselSize: "+str(ReselSize))
print("Volume: "+str(Volume))
print("ReselCount: "+str(ReselCount))
print("------------")

# From ReselCount to FWE treshold

FweThres_cmd = 'ptoz 0.05 -g %s' %ReselCount
FweThres = os.popen(FweThres_cmd).read()

print("FWE voxelwise GRF threshold: "+str(FweThres))

Power = 0.8

muRange = np.arange(1.8,5,0.01)
muSingle = []
for muMax in muRange:
# what is the power to detect a maximum 
    power = 1-integrate.quad(lambda x:peakdistribution.peakdens3D(x,1),-20,float(FweThres)-muMax)[0]
    if power>Power:
        muSingle.append(muMax)
        break
print("The power is sufficient for one region if mu equals: "+str(muSingle[0]))

# Read in data
Data = pd.read_csv("../SampleSize/neurosynth_study_data.txt",sep=" ",header=None,names=['year','n'])
Data['source']='Tal'
Data=Data[Data.year!=1997] #remove year with 1 entry
David = pd.read_csv("../SampleSize/david_sampsizedata.txt",sep=" ",header=None,names=['year','n'])
David['source']='David'
Data=Data.append(David)

# add detectable effect
Data['deltaSingle']=muSingle[0]/np.sqrt(Data['n'])

# add jitter for figure
stdev = 0.01*(max(Data.year)-min(Data.year))
Data['year_jitter'] = Data.year+np.random.randn(len(Data))*stdev

# Compute medians per year (for smoother)
Medians = pd.DataFrame({'year':
                     np.arange(start=np.min(Data.year),stop=np.max(Data.year)+1),
                      'TalMdSS':'nan',
                     'DavidMdSS':'nan',
                      'TalMdDSingle':'nan',
                     'DavidMdDSingle':'nan',
                        'MdSS':'nan',
                        'DSingle':'nan'
                       })

for yearInd in (range(len(Medians))):
    # Compute medians for Tal's data
    yearBoolTal = np.array([a and b for a,b in zip(Data.source=="Tal",Data.year==Medians.year[yearInd])])
    Medians.TalMdSS[yearInd] = np.median(Data.n[yearBoolTal])
    Medians.TalMdDSingle[yearInd] = np.median(Data.deltaSingle[yearBoolTal])
    # Compute medians for David's data
    yearBoolDavid = np.array([a and b for a,b in zip(Data.source=="David",Data.year==Medians.year[yearInd])])
    Medians.DavidMdSS[yearInd] = np.median(Data.n[yearBoolDavid])
    Medians.DavidMdDSingle[yearInd] = np.median(Data.deltaSingle[yearBoolDavid])
    # Compute medians for all data
    yearBool = np.array(Data.year==Medians.year[yearInd])
    Medians.MdSS[yearInd] = np.median(Data.n[yearBool])
    Medians.DSingle[yearInd] = np.median(Data.deltaSingle[yearBool])   

Medians[0:5]

# add logscale

Medians['MdSSLog'] = [np.log(x) for x in Medians.MdSS]
Medians['TalMdSSLog'] = [np.log(x) for x in Medians.TalMdSS]
Medians['DavidMdSSLog'] = [np.log(x) for x in Medians.DavidMdSS]

Data['nLog']= [np.log(x) for x in Data.n]

twocol = cb.qualitative.Paired_12.mpl_colors

fig,axs = plt.subplots(1,2,figsize=(12,5))
fig.subplots_adjust(hspace=.5,wspace=.3)
axs=axs.ravel()
axs[0].plot(Data.year_jitter[Data.source=="Tal"],Data['nLog'][Data.source=="Tal"],"r.",color=twocol[0],alpha=0.5,label="")
axs[0].plot(Data.year_jitter[Data.source=="David"],Data['nLog'][Data.source=="David"],"r.",color=twocol[2],alpha=0.5,label="")
axs[0].plot(Medians.year,Medians.TalMdSSLog,color=twocol[1],lw=3,label="Neurosynth")
axs[0].plot(Medians.year,Medians.DavidMdSSLog,color=twocol[3],lw=3,label="David et al.")
axs[0].set_xlim([1993,2016])
axs[0].set_ylim([0,8])
axs[0].set_xlabel("Year")
axs[0].set_ylabel("Median Sample Size")
axs[0].legend(loc="upper left",frameon=False)
#labels=[1,5,10,20,50,150,500,1000,3000]
labels=[1,4,16,64,256,1024,3000]
axs[0].set_yticks(np.log(labels))
axs[0].set_yticklabels(labels)


axs[1].plot(Data.year_jitter[Data.source=="Tal"],Data.deltaSingle[Data.source=="Tal"],"r.",color=twocol[0],alpha=0.5,label="")
axs[1].plot(Data.year_jitter[Data.source=="David"],Data.deltaSingle[Data.source=="David"],"r.",color=twocol[2],alpha=0.5,label="")
axs[1].plot(Medians.year,Medians.TalMdDSingle,color=twocol[1],lw=3,label="Neurosynth")
axs[1].plot(Medians.year,Medians.DavidMdDSingle,color=twocol[3],lw=3,label="David et al.")
axs[1].set_xlim([1993,2016])
axs[1].set_ylim([0,3])
axs[1].set_xlabel("Year")
axs[1].set_ylabel("Effect Size with 80% power")
axs[1].legend(loc="upper right",frameon=False)
plt.savefig('Figure1.svg',dpi=600)
plt.show()


Medians.loc[Medians.year>2010, lambda df: ['year', 'TalMdSS', 'TalMdDSingle']]

yearBoolTal = np.array([a and b for a,b in zip(Data.source=="Tal",Data.year>2010)])
print('Median sample size (2011-2015):',np.median(Data.n[yearBoolTal]))

for year in range(2011,2016):
    bigstudyBoolTal = np.array([a and b and c for a,b,c in zip(Data.source=="Tal",Data.n>100,Data.year==year)])
    yearBoolTal = np.array([a and b for a,b in zip(Data.source=="Tal",Data.year==year)])


    print(year,np.sum(yearBoolTal),np.sum(Data[bigstudyBoolTal].n>99))

groupData = pd.read_csv("../SampleSize/neurosynth_group_data.txt",sep=" ",header=None,names=['year','n'])
yearBoolGroup=np.array([a for a in groupData.year==year])
print('%d groups found in 2015 (over %d studies)'%(np.sum(yearBoolGroup),np.unique(groupData[yearBoolGroup].index).shape[0]))
print('median group size in 2015: %f'%np.median(groupData[yearBoolGroup].n))



