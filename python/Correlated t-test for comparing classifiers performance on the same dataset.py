import numpy as np
Acc_nbc =  np.loadtxt('Data/nbc_anneal.csv',  delimiter=',', skiprows=1)
Acc_aode = np.loadtxt('Data/aode_anneal.csv', delimiter=',', skiprows=1)
names = ("NBC", "AODE")
x=np.zeros((len(Acc_nbc),2),'float')
x[:,1]=Acc_nbc/100
x[:,0]=Acc_aode/100
#we consider the difference of accuracy scaled in (0,1)

import bayesiantests as bt
rope=0.01
left, within, right = bt.correlated_ttest(x, rope=rope,runs=10,verbose=True,names=names)

import warnings
warnings.filterwarnings('ignore')
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as snb
#generate samples from posterior (it is not necesssary because the posterior is a Student)
samples=bt.correlated_ttest_MC(x, rope=rope,runs=10,nsamples=50000)
#plot posterior
snb.kdeplot(samples, shade=True) 
#plot rope region
plt.axvline(x=-rope,color='orange')
plt.axvline(x=rope,color='orange')
#add label
plt.xlabel('Nbc-Aode on Anneal dataset') 



import numpy as np
Acc_nbc =  np.loadtxt('Data/nbc_audiology.csv',  delimiter=',', skiprows=1)
Acc_aode = np.loadtxt('Data/aode_audiology.csv', delimiter=',', skiprows=1)
names = ("NBC", "AODE")
diff=(Acc_nbc-Acc_aode)/100.0 #we consider the difference of accuracy scaled in (0,1)

import bayesiantests as bt
rope=0.01
left, within, right = bt.correlated_ttest(diff, rope=rope,runs=10,verbose=True,names=names)

import warnings
warnings.filterwarnings('ignore')
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as snb
#generate samples from posterior (it is not necesssary because the posterior is a Student)
samples=bt.correlated_ttest_MC(diff, rope=rope,runs=10,nsamples=50000)
#plot posterior
snb.kdeplot(samples, shade=True) 
#plot rope region
plt.axvline(x=-rope,color='orange')
plt.axvline(x=rope,color='orange')
#add label
plt.xlabel('Nbc-Aode on Audiology dataset') 



