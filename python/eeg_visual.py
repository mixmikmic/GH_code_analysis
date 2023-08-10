from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as pp
#from sklearn.preprocessing import normalize

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    print(norm)
    return v/norm

data = genfromtxt('../test_data/shouvik1602/shouvik1602.csv', delimiter=',', dtype = np.float32)
print('DATA SHAPE:', data.shape)
data[0][:]

data[0][:] 
data = data[1::][:]

fp2 = data[:,[1]]  
fp1 = data[:,[0]]

pp.figure(figsize=(30,10))
pp.plot(fp2)
pp.show()

pp.figure(figsize=(30,10))
pp.plot(fp1)
pp.show()

#n_fp1 = normalize(fp1)
#n_fp2 = normalize(fp2)

print('FP1 peak =',np.max(fp1))
print('FP2 peak =',np.max(fp2))
n_fp1 = fp1/np.max(fp1)
n_fp2 = fp2/np.max(fp2)

pp.figure(figsize=(30,10))
pp.plot(fp2)
pp.show()

pp.figure(figsize=(30,10))
pp.plot(fp1)
pp.show()



