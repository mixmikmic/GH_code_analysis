import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')
plt.style.use('notebook');
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

π = np.pi
print(π)

from IPython.display import HTML
HTML('<iframe src=http://matplotlib.org/users/pyplot_tutorial.html#controlling-line-properties width=700 height=350></iframe>')

# For all the details
HTML('<iframe src=http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot width=700 height=350></iframe>')

theta = np.linspace(0,2*np.pi,20)
sinlabel = r'$\sin(\theta)$'
coslabel = r'$\cos(\theta)$'
plt.figure(figsize=(8,6))
plt.plot(theta,np.sin(theta),label=sinlabel,color=(0.0,0.0,0.6),linestyle='-',linewidth=2.5,marker='o',markeredgecolor='green',    markerfacecolor='#0FDDAF',markersize=12.0,markeredgewidth=0.8)
plt.plot(theta,np.cos(theta),label=coslabel,color='red',linestyle='None',         linewidth=0.5,marker='s',markeredgecolor='None',         markerfacecolor='pink',markersize=15.0)
plt.xlabel(r'$\theta$')
plt.xlim(0,2*np.pi)
plt.legend(loc='lower left', fontsize=15)

x = np.linspace(-1,1,20)
colors = ["#2078B5", "#FF7F0F", "#2CA12C", "#D72827", "#9467BE", "#8C574B",
            "#E478C2", "#808080", "#BCBE20", "#17BED0", "#AEC8E9", "#FFBC79", 
            "#98E08B", "#FF9896", "#C6B1D6", "#C59D94", "#F8B7D3", "#C8C8C8", 
           "#DCDC8E", "#9EDAE6"]
marker_list = ['o','s','p','^','v','>','<','D','*','H']
for i,marker in enumerate(marker_list):
    n = len(marker_list)
    color = ((i+1.0)/n,0,(n-i)/n)
    plt.plot(x,x**2+i,marker=marker, color=color, markerfacecolor=color,linewidth=1, markersize=15)



