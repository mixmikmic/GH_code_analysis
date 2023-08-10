import numpy as np
from time import time
import pandas as pd
from scipy.stats import skew,kurtosis

loader = pd.read_pickle("./data.pkl")
loader = np.array(loader)
line = np.random.choice(range(9120),size=100,replace=False)
data = loader[line]


def get_features(a):
    a=np.array(a)
    mean_data=np.mean(a,axis=0)
    dev=np.std(a,axis=0)
    corrln=[]
    for i in xrange(0,a.shape[1],3):
        corrln.append(np.corrcoef(a[:,i],a[:,i+1])[1,0])
        corrln.append(np.corrcoef(a[:,i],a[:,i+2])[1,0])
        corrln.append(np.corrcoef(a[:,i+1],a[:,i+2])[1,0])
    to_send=mean_data
    to_send=np.concatenate((to_send,dev))
    to_send=np.concatenate((to_send,corrln))
    to_send=np.concatenate((to_send,np.min(a,axis=0)))
    to_send=np.concatenate((to_send,np.max(a,axis=0)))
    to_send=np.concatenate((to_send,skew(a,axis=0)))
    to_send=np.concatenate((to_send,kurtosis(a,axis=0)))
    return 
    

def time_feat(k):
    l = np.zeros((100,))
    for i in xrange(100):
        t = time()
        get_features(k[i])
        tt = time()-t
        l[i] = tt
    return l

np.mean(l)

time_acc = time_feat(data)

acc_0 = data[:,:,0:3]
acc_1 = data[:,:,3:6]
acc_2 = data[:,:,6:9]
acc_3 = data[:,:,9:12]
acc_4 = data[:,:,12:15]

time_acc0 = time_feat(acc_0)
time_acc1 = time_feat(acc_1)
time_acc2 = time_feat(acc_2)
time_acc3 = time_feat(acc_3)
time_acc4 = time_feat(acc_4)

print np.mean(time_acc)

print np.mean(time_acc0)

print np.mean(time_acc1)

print np.mean(time_acc2)

print np.mean(time_acc3)

print np.mean(time_acc4)

data = pd.read_pickle("./data.pkl")
data = np.array(data)
print data.shape

def time_featy(k,lines):
    l = np.empty((1,100))
    for i in lines:
        t = time()
        get_features(k[i])
        tt = time()-t
        l = np.append(l,tt)
    return l

def time_taken(d):
    tt = np.array((0,))
    lines = np.random.choice(480,size=100,replace=False)
    for i in lines:
        start = time()
        get_features(d[i])
        tt = np.append(tt,time()-start)
    tt = np.delete(tt,0)
    acc_0 = d[:,:,0:3]
    acc_1 = d[:,:,3:6]
    acc_2 = d[:,:,6:9]
    acc_3 = d[:,:,9:12]
    acc_4 = d[:,:,12:15]
    time_acc0 = time_featy(acc_0,lines)
    time_acc1 = time_featy(acc_1,lines)
    time_acc2 = time_featy(acc_2,lines)
    time_acc3 = time_featy(acc_3,lines)
    time_acc4 = time_featy(acc_4,lines)
    print "Mean time for acc0: ",np.mean(time_acc0)
    print "Mean time for acc1: ",np.mean(time_acc1)
    print "Mean time for acc2: ",np.mean(time_acc2)
    print "Mean time for acc3: ",np.mean(time_acc3)
    print "Mean time for acc4: ",np.mean(time_acc4)
    return tt

act = np.empty((19,480,125,15))
time_for_act = np.empty((19,100))

for i in xrange(19):
    print "Activity ",i+1
    act[i] = data[480*(i):(i+1)*480,:,:]
    time_for_act[i] = time_taken(act[i])
    print "Mean time for all acc: ",np.mean(time_for_act[i])

k = np.empty((1,19))
for i in xrange(19):
    k = np.append(k,np.mean(time_for_act[i],axis=0))
    print "Act",i,":",np.mean(time_for_act[i],axis=0)
print 
print "Mean time for all 19 act: ",np.mean(k)



