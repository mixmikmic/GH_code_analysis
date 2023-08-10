import numpy as np
import matplotlib.pyplot as plt

# 8-component leakage function; beta is a 9-element array
beta9 = np.load('results/lrmodel9_1000traces.npy') # 9 coefs, the last one being the intercept
def leakageModel_LR9(x):
    result = beta9[8]
    for i in range(0, 8):
        bit = (x >> i) & 1  # this is the definition: gi = [bit i of x]
        result += beta9[i] * bit
    return result

# Hamming weight leakage model
byteHammingWeight = np.load('../data/bytehammingweight.npy') # HW table
def leakageModel_HW(x):
    return byteHammingWeight[x]

# templates (reduced, i.e. just the mean, because correlation distinguisher does not use variance)
means = np.load('results/means1000.npy') # HW table
def leakageModel_T(x):
    return means[x]

# SboxNum and SampleNum should be the same as for building the leakage models
inputRange = range(1000, 2000) # range for traces (not samples!)
SboxNum = 0
SampleNum = 1025

# load samples and data
npzfile = np.load('../traces/swaes_atmega_power.npz')
data = npzfile['data'][inputRange,SboxNum]
traces = npzfile['traces'][inputRange,SampleNum]

# known key, such that we can highlight the corresponding correlation trace
key = b'\x2b\x7e\x15\x16\x28\xae\xd2\xa6\xab\xf7\x15\x88\x09\xcf\x4f\x3c'
knownKeyByte = np.uint8(key[SboxNum])

# load AES S-Box
sbox = np.load('../data/aessbox.npy')

# compute intermediate variable hypotheses for all the key candidates
k = np.arange(0,256, dtype='uint8')
H = np.zeros((256, len(data)), dtype='uint8')
for i in range(256):
    H[i,:] = sbox[data ^ k[i]]

# apply leakage function to intermediate variables to get leakage predictions
# we do it for all models for comparison
HL_HW = list(map(leakageModel_HW, H))
HL_LR = list(map(leakageModel_LR9, H))
HL_T  = list(map(leakageModel_T, H))

startTrace = 0  # trace to start with 
n = range(10,100) # trace numbers through which to run

# arrays for correlation evlution
Correlation_HW = []
Correlation_LR = []
Correlation_T = []

for i in range(256): # for each key candidate
    a = np.zeros(len(n))
    for j in range(len(n)):
        a[j] = np.corrcoef(traces[startTrace:startTrace+n[j]], HL_HW[i][startTrace:startTrace+n[j]])[0,1]
    Correlation_HW.append(a)
    
    b = np.zeros(len(n))
    for j in range(len(n)):
        b[j] = np.corrcoef(traces[startTrace:startTrace+n[j]], HL_LR[i][startTrace:startTrace+n[j]])[0,1]
    Correlation_LR.append(b)
    
    c = np.zeros(len(n))
    for j in range(len(n)):
        c[j] = np.corrcoef(traces[startTrace:startTrace+n[j]], HL_T[i][startTrace:startTrace+n[j]])[0,1]
    Correlation_T.append(c)
    
fig, (axhw, axlr, axt) = plt.subplots(3,1, sharex=True, figsize=(10, 10), squeeze=True)
for i in range(256):
    axhw.plot(n, Correlation_HW[i], color='silver')
    axlr.plot(n, Correlation_LR[i], color='silver')
    axt.plot(n, Correlation_T[i], color='silver')
axhw.plot(n, Correlation_HW[knownKeyByte], color='red') # highlihght the correct key byte candidates
axlr.plot(n, Correlation_LR[knownKeyByte], color='red')
axt.plot(n, Correlation_T[knownKeyByte], color='red')
axhw.set_ylim([0,1]) # negative value are not needed in this case
axlr.set_ylim([0,1])
axt.set_ylim([0,1])
axhw.set_ylabel("corr HW")
axlr.set_ylabel("corr LR")
axt.set_ylabel("corr T")
axt.set_xlabel("Nubmer of traces in the attack")
plt.show()

