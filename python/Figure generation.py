import matplotlib.pylab as plt
import numpy as np
from pickle import load
import pymc3

inFile = open('trace_n100_genParams.pkl','rb')
trace = load(inFile)
inFile.close()

datadir = '../data/small_sample/'
infile = open(datadir+'T.pkl','rb')
T = load(infile)
infile.close()
infile = open(datadir+'obs_jumps.pkl','rb')
obs_jumps = load(infile)                                                                                                                                                                          
infile.close()
infile = open(datadir+'O.pkl','rb')
O = load(infile)
infile.close()

newN = 100
T = T[:newN]
nObs = T.sum()
obs_jumps = obs_jumps[0:nObs] 
O = O[0:nObs]

pi = trace['pi']
Q = trace['Q']
S = trace['S']
B0 = trace['B0']
B = trace['B']
X = trace['X']
Z = trace['Z']
L = trace['L']

N = T.shape[0] # Number of patients
M = pi[0].shape[0] # Number of hidden states                                                                                                                                                   
K = Z[0].shape[0] # Number of comorbidities
D = Z[0].shape[1] # Number of claims

Sbin = np.vstack([np.bincount(S[i],minlength=4)/float(len(S[i])) for i in range(len(S))])
zeroIndices = np.roll(T.cumsum(),1)
zeroIndices[0] = 0
pibar = np.vstack([np.bincount(S[i][zeroIndices],minlength=M)/float(zeroIndices.shape[0]) for i in range(len(S))])
SEnd = np.vstack([np.bincount(S[i][zeroIndices-1],minlength=M)/float(zeroIndices.shape[0]) for i in range(len(S))])
XChanges = np.insert(1-(1-(X[:,1:]-X[:,:-1])).prod(axis=2),0,0,axis=1)
XChanges.T[zeroIndices] = 0
XChanges[XChanges.nonzero()] = XChanges[XChanges.nonzero()]/XChanges[XChanges.nonzero()]
XChanges = XChanges.sum(axis=1)/float(N)

nPick=10
tPick = 1
ntStart = zeroIndices[nPick]
curObs = ntStart + tPick+1

#print obs_jumps[ntStart:ntStart+10]
#print O[ntStart:ntStart+10]
#print X[-1][ntStart:ntStart+10]



#OK, do this properly later

likelihood_S_0_1 = np.array([9.53625347e-01,   1.86315267e-02,   1.44532352e-03, 3.14356015e-04])

likelihood_X_0_1 = np.array([[  2.56550334e-14,   1.21956409e-04],

       [ 1.63014965e-04,1.63011786e-04],

       [  1.42531099e-04, 4.28589290e-05],

       [  1.27941508e-09,9.56828390e-05]])
likelihood_X_0_1 = likelihood_X_0_1[:,1]/likelihood_X_0_1.sum(axis=1)

print likelihood_S_0_1
print likelihood_X_0_1



