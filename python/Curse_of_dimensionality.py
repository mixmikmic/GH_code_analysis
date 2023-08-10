from IPython.display import Image 
Image('../../../python_for_probability_statistics_and_machine_learning.jpg')

from __future__ import division
get_ipython().magic('pylab inline')

# uniformly in unit-cube
from matplotlib.patches import Circle
v=np.random.rand(1000,2)-1/2.
fig,ax=subplots()
ax.set_aspect(1)
ax.scatter(v[:,0],v[:,1],color='gray',alpha=.3)
ax.add_patch(Circle((0,0),0.5,alpha=.8,lw=3.,fill=False))


siz = [2,3,5,10,20,50,100,500,1000]
siz = siz[:6]
print sqrt(v.shape[1])/2./2.

fig,axs=subplots(3,2)
fig.set_size_inches((8,5))
for ax,k in zip(axs.flatten(),siz):
    v=np.random.rand(5000,k)-1/2
    ax.hist([np.linalg.norm(i) for i in v],color='gray',normed=True);
    ax.vlines(0.5,0,ax.axis()[-1],lw=3)
    ax.set_title('$d=%d$'%k,fontsize=20)
fig.set_tight_layout(True)

