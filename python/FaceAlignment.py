#load the test dataset 
import pandas as pd
import numpy as np
import pylab as plt
import CAMFA as CF ##this is the python class used for evaluation
import prettyplotlib as ppl
import matplotlib
import facedraw as fd##functions used to draw result figures in the paper 
import pickle as pkl
dbtest = pd.read_pickle('./dataset/CAM300W_test.pkl')
dbtest.head(3)

#an example of using CAMFA
#initalise the alignment evaluation class
get_ipython().magic('matplotlib inline')
gtxy = np.array([xy for xy in dbtest['GTLMs'].values])# get ground truth xy
GEA = CF.Group_Error_ANA(gtxy)# initialise group error analyser 
#assume a file called 'result.txt' stores the face alignment result of a certain method and the format is correct 
result = np.loadtxt('result.txt') 
print('Result shape is {}'.format(result.shape))
auc, bins, CDF = GEA.get_edf_histogram(result,thr_=0.2)#thr_ is the alpha value (0.3 by defualt)

fig = plt.figure()
ax = fig.add_subplot(111)
ppl.plot(bins,CDF,lw=3)
ppl.fill_between(bins,0*len(CDF),CDF,alpha=0.5)
plt.xlabel('Normalised error')
plt.ylabel('Proportion of landmarks')
an = ax.annotate('AUC=%1.4f'%auc,xy=(0.13,0.5))

fd.basic_compare()

matplotlib.rcParams['figure.figsize'] = (9, 7)
fd.draw_ots_sens_center()

fd.draw_ots_sens_scale()

matplotlib.rcParams['figure.figsize'] = (13, 4)
fd.draw_real_facebb_res()



