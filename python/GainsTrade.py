import seaborn as sns
import matplotlib.pyplot as plt
from ipywidgets import interact

get_ipython().magic('matplotlib inline')
from matplotlib import rcParams
rcParams['figure.figsize'] = (6, 6)

LE = 10    
MPL_F = 1.0
MPL_C = 1.0

def ppf(mplF = 1, mplC=1, LE = LE, wp =1, price=False):
    QFmax = mplF * LE
    QCmax = mplC * LE
    print('Domestic relative price:')
    print(' {:5.2f} coconuts per Fish'.format(mplC/mplF))
    plt.plot([0, QFmax], [ QCmax,0 ], marker='o',clip_on=False)
    
    if price:
        print("At a world price of {} coconut per fish this country:".format(wp))
        if (wp < mplC/mplF):
            plt.plot([0, QCmax*wp], [QCmax,0 ], 'r--',clip_on=False)
            print('   exports COCONUTS')
        elif (wp > mplC/mplF):
            plt.plot([0, QFmax], [QFmax*wp,0 ], 'r--',clip_on=False)
            print('   exports FISH')
        else:
            print('   has NO TRADE')
        
    plt.title("Production Possibility Frontier", fontsize = 18)
    plt.xlim(0, LE*2) 
    plt.ylim(0, LE*2)
    plt.xlabel("QF -- Fish")
    plt.ylabel("QC -- Coconuts")
    plt.show()

ppf(2,1)

interact(ppf, mplF=(0.1, 3, 0.1), mplC=(0.1, 3, 0.1), LE=(1,2*LE,LE/10),wp=(0.1,3,0.1))





