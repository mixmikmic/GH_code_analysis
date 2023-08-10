import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

get_ipython().magic('matplotlib inline')

data_dir = "data/cokol_et_al_2011/"
data_files = os.listdir(data_dir)
data_files.remove("README.txt")

data_files[0]

data = pd.read_csv(os.path.join(data_dir,data_files[0]),sep="\t",header=None)
# data = pd.read_csv(os.path.join(data_dir,'Bro-Bro.txt'),sep="\t",header=None)
data = np.log2(data)
data = data.iloc[4:,:]
data = data - data.iloc[0,:]
data.shape

data.columns

def column_concentrations(ind):
    return 1./8*(ind/8),1.*ind%8/8

concs = [column_concentrations(i) for i in data.columns]
concs = pd.DataFrame(concs)
concs

plt.figure(figsize=(20,20))

ylim = (data.min().min(),data.max().max())

for i in range(8):
    for j in range(8):
        num = i%8+j*8
        c1 = 1.*num/8
        c2 = 1.*num%8/8
        plt.subplot(8,8,num+1)
        plt.plot(data.iloc[:,num])
        plt.ylim(ylim)
        #plt.title("%d, %d, %d"%(i,j,num))
        plt.title("%.2lf, %.2lf"%(c1,c2))
    
plt.tight_layout()

concs[0].factorize()

concs[0]

column_concentrations(4)



