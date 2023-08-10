get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

ALPHA = 0.5
BETA = 0.7
TBAR = 100
LBAR = 100

def F(T,L,alpha=ALPHA):
    return (T**alpha)*(L**(1-alpha))

def FT(T,L,alpha=ALPHA):
    """Shadow price of labor"""
    return alpha*F(T,L,alpha=ALPHA)/T

def yao_carter(Ti, rO, rI, alpha=ALPHA):
    """returns optimal land use and shadow rental price of land"""
    r = FT(Ti, LBAR, alpha)
    Tout = LBAR * (alpha/rO)**(1/(1-alpha))
    Tin = LBAR * (alpha/rI)**(1/(1-alpha))
    TD = (r < rO)*Tout + (r > rI)* Tin  + ((r>=rO) & (r<=rI))*Ti
    rs = (r < rO)*rO + (r > rI)* rI  + ((r>=rO) & (r<=rI)) * r
    fig, ax1 = plt.subplots(figsize=(8,6))
    ax1.plot(Ti,TD,label='land use')
    ax1.set_title("Three land regimes")
    ax1.set_xlabel('Land endowment '+r'$\bar T_e$')
    ax1.set_ylabel('Farm size (land use)')
    ax1.grid()
    ax2 = ax1.twinx()
    ax2.plot(Ti, rs,'k--',label='shadow price land')
    ax2.set_ylabel('Shadow Price of land')
    ax2.set_ylim(0.5,0.85)
    ax1.axvspan(Tin, Tout, alpha=0.2, color='red')
    legend = ax1.legend(loc='upper left', shadow=True)
    legend = ax2.legend(loc='lower right', shadow=True)
    plt.show()

Ti = np.linspace(1,TBAR,num=100)
RI = 0.8
RO = 0.6
yao_carter(Ti, RO, RI, alpha=ALPHA)

