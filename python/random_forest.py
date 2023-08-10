import numpy as np
import matplotlib.pyplot as plt
import math
plt.rc('font', size=20)          # controls default text sizes
plt.rc('axes', titlesize=16)     # fontsize of the axes title
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels   # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels

def H(fp):
    fn = 1.0 - fp
    H = -1.0 * fp * math.log(fp,2) - fn * math.log(fn,2)
    return H

fp_ = np.arange(0.01,1,0.01)
H_ = [ H(fp) for fp in fp_ ]
plt.figure(figsize=(10,6))
plt.plot(fp_, H_)
plt.xlabel('$p^{+}$')
plt.ylabel('Entropy')
plt.show()



