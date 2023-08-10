import io
import urllib.request
import numpy as np

url = "https://ndownloader.figshare.com/files/9956302"
response = urllib.request.urlopen(url)
data = response.read()
bytes = io.BytesIO(data)
topo = np.load(bytes)

import numpy as np
topo = np.load("./OneTreeHillTopo.npy") 

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
from matplotlib import cm
fig = plt.figure()
#plt.axis('off')
plt.imshow(topo,cmap=cm.gist_earth)
cbar = plt.colorbar(shrink=0.9)
cbar.set_label('Altitude above sea level (m)')
plt.show()

fy,fx = np.gradient(topo,50)

ymax,xmax = fx.shape
fig = plt.figure()
plt.axis('off')
plt.quiver(np.arange(xmax),np.arange(ymax),fx,-fy)
plt.imshow(topo,cmap=cm.gist_earth)
cbar = plt.colorbar(shrink=0.9)
cbar.set_label('Altitude above sea level (m)')
plt.show()

fig = plt.figure()
plt.axis('off')
plt.imshow(np.sqrt(fx**2+fy**2),cmap=cm.gist_earth)
cbar = plt.colorbar(shrink=0.9)
cbar.set_label('|gradient| (dimensionless)')
plt.axis('off')
plt.show()



