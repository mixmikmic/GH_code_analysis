get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [12,8]

def Shor_vs_GNFS(X_min,X_max, scale='linear', Shor_only=0, GNFS_only=0):
    mu = np.linspace(X_min, X_max, 800)
    GNFS = np.exp(mu**(1/3)*(np.log2(mu))**2/3)
    Shor = mu**3
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    if scale == 'log':
        ax.set_yscale('log')
    if not GNFS_only:
        plt.plot(mu, Shor, linewidth=3, label=r"Shor's Algorithm")
    if not Shor_only:   
        plt.plot(mu, GNFS, linewidth=3, label=r"GNFS")
    plt.xlabel(r'Integer Bit-Width')
    plt.ylabel(r'Order of Required Process Steps')
    plt.legend(loc=0)
    plt.show()

Shor_vs_GNFS(2,10,scale='linear')

Shor_vs_GNFS(2,20,scale='linear')

Shor_vs_GNFS(200, 1200,scale='log')

Shor_vs_GNFS(1110, 1112, GNFS_only=1)
Shor_vs_GNFS(1110, 1112, Shor_only=1)



