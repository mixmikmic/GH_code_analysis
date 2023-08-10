import matplotlib.pyplot as plt
import numpy as np

get_ipython().magic('matplotlib inline')
plt.style.use('notebook');
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
colors = ["#2078B5", "#FF7F0F", "#2CA12C", "#D72827", "#9467BE", "#8C574B",
            "#E478C2", "#808080", "#BCBE20", "#17BED0", "#AEC8E9", "#FFBC79", 
            "#98E08B", "#FF9896", "#C6B1D6", "#C59D94", "#F8B7D3", "#C8C8C8", 
           "#DCDC8E", "#9EDAE6"]

M = np.linspace(-1.5,1.5,1000)

plt.plot(M,M,color='k', label='$y=M$')
for cx in [2,1.25,1.0,0.5]:
    plt.plot(M,np.tanh(M*cx), label='x = %3.2f' % cx)
plt.xlabel(r'$M$')
plt.legend(loc='lower right')
plt.title('Mean Field Theory')

def mean_field_eqn(m,cx):
    '''The mean field equation for the magnetization.'''
    return m - np.tanh(m*cx)

from scipy.optimize import fsolve
x = np.linspace(0.5,100,10000)
Mx = [fsolve(mean_field_eqn, 1.1, args=(cx)) for cx in x]

plt.plot(1.0/x,Mx, linewidth=4)
plt.xlabel('Temperature  $T/zJ$')
plt.ylabel('Magnetization  $M$')
plt.ylim(-0.001,1.05);



