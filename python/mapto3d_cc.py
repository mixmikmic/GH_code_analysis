import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from IPython.core.debugger import set_trace

# convert image to code
img = mpimg.imread("Picture1.png")

# save image as pickle
pickle_name = "Picture1.p"
pickle.dump(img, open(pickle_name, "wb"))

# load pickle
img1 = pickle.load(open(pickle_name, "rb"), encoding='latin1')

# converte image to binary
lum_img1 = img1[:,:,0]

# pixel dimensions
px = lum_img1.shape[0]
py = lum_img1.shape[1]
print('img1 is', px,'pixels by', py, 'pixels')

x = [[] for i in range(9)]
y = [[] for i in range(9)]
z = [[] for i in range(9)]

threshold = 0.90

for i in range(9):
    for j in range(py):
        for k in range(px):
            if lum_img1[j,k] < threshold:
                x[i].append(k)
                y[i].append(j)
    z[i]= (i+1) * np.ones(np.size(x[i]))
    threshold = threshold - 0.10

#print(np.size(x[2]),len(y[2]),len(z[2]))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(z[0],x[0],y[0],'o',color='magenta')
ax.plot(z[1], x[1], y[1],'o',color='yellow')
ax.plot(z[2], x[2], y[2],'o',color='black')
ax.plot(z[3], x[3], y[3],'o',color='cyan')
ax.plot(z[4],x[4], y[4],'o',color='pink')
ax.plot(z[5],x[5], y[5],'o',color='magenta')
ax.plot(z[6],x[6], y[6],'o',color='blue')
ax.plot(z[7],x[7], y[7],'o',color='navy')
ax.plot(z[8],x[8], y[8],'o',color='black')
plt.xlabel('x')
plt.ylabel('y')
# rotate the axes
ax.view_init(160, 315)





