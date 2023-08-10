get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pylab as pylab
import seaborn as sns

#Plot formatting for presentation
plt.style.use(['bmh'])

params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

#Load the header information from the text file
#The header contains the number of signal and background events
mbsigback=pd.read_table("MiniBooNE_PID.txt", delimiter=r"\s+", nrows=1, header=None)
nsig=mbsigback[0][0]
nback=mbsigback[1][0]

#Load the signal and background into separate dataframes
mbsig=pd.read_table("MiniBooNE_PID.txt", delimiter=r"\s+", nrows=1000, skiprows=1, header=None)
mbback=pd.read_table("MiniBooNE_PID.txt", delimiter=r"\s+", nrows=1000, skiprows=1+nsig, header=None)

#Add labels column to mbsig and mbback (0 = signal, 1 = background)
mbsig['sigback'] = pd.Series([0 for x in range(len(mbsig.index))], index=mbsig.index)
mbback['sigback'] = pd.Series([1 for x in range(len(mbback.index))], index=mbback.index)

#Merge the dataframes into one
mball=pd.concat([mbsig,mbback]) 

#Clean default values from the dataset (remove entries with -999.00 in any field)
for col in mball:
    mball=mball[mball[col]!=-999.00]
    
#Rescale data to be from 0 to 1
norm_cols = [i for i in range(50)]
mball[norm_cols] = mball[norm_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

print("Total signal entries:", nsig)
print("Total background entries:", nback)


mbsig.describe()

mbback.describe()

mball.describe()

#Do some plotting
for col in range(50):
    plt.figure()
    #Extract signal and background for this column from pandas dataframe
    sig=mball[mball.sigback==0][col]
    back=mball[mball.sigback==1][col]
    #Get binning by merging the datasets, plotting, and returning the binning (index 1)
    bins=np.histogram(np.hstack((sig,back)), bins=40)[1]
    plt.hist(sig, label='Signal', alpha=0.5, bins=bins)
    plt.hist(back, label='Background', alpha=0.5, bins=bins)
    plt.xlabel('Feature %d'%col)
    plt.legend()
    plt.show()

#Plot pairwise 2d correlation plots for the first 5 features
#Note: this uses the seaborn library
sns.pairplot(mball, hue="sigback", vars=[0,1,2,3,4], size=2.5, plot_kws=dict(s=10))

