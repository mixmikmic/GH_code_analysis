import os, glob, cv2, imutils
import numpy as np
import matplotlib.pyplot as plt
from util import get_dataset
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
print ("Packages Loaded.")

datasetname = 'face_emotion'
loadpath = '../data/face_emotion/'
rszshape = (64,64)
labels   = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
print ("Load Configuration Ready")

""" Load (Shuffle with Fixed Seed) """
X,Y,imgcnt = get_dataset(_loadpath=loadpath,_rszshape=rszshape,_imgext='png',_VERBOSE=True)
X = X / 255.
""" Divide into Train / Test / Validation """
trainimg,trainlabel = X[:int(imgcnt*0.7),:],Y[:int(imgcnt*0.7),:]
testimg,testlabel = X[int(imgcnt*0.7):int(imgcnt*0.9),:],Y[int(imgcnt*0.7):int(imgcnt*0.9),:]
valimg,vallabel = X[int(imgcnt*0.9):,:],Y[int(imgcnt*0.9):,:]
print ("#Train:[%d], #Test:[%d], #Validation:[%d]"%
       (trainimg.shape[0],testimg.shape[0],valimg.shape[0]))

f,axarr = plt.subplots(1,5,figsize=(18,8))
for idx,imgidx in enumerate(np.random.randint(X.shape[0],size=5)):
    currimg=np.reshape(X[imgidx,:],rszshape)
    currlabel=labels[np.argmax(Y[imgidx,:])]
    axarr[idx].imshow(currimg,cmap=plt.get_cmap('gray'))
    axarr[idx].set_title('[%d] %s'%(imgidx,currlabel),fontsize=15)

savepath = '../data/'+datasetname+'.npz'
np.savez(savepath,X=X,Y=Y,rszshape=rszshape,labels=labels,
         trainimg=trainimg,trainlabel=trainlabel,testimg=testimg,testlabel=testlabel,
         valimg=valimg,vallabel=vallabel,imgcnt=imgcnt)
print ("[%s] Saved."%(savepath))
print('Size is [%.1f]MB'%(os.path.getsize(savepath)/1000./1000.)) 

