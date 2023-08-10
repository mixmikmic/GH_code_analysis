import numpy as np
import matplotlib.pyplot as plt

# defining the inner square as 3x3 array with an initial gray value
inner_gray_value = 120
inner_square = np.full((3,3), inner_gray_value, np.double)

# defining the outer squares and overlaying the inner square
a = np.zeros((5,5), np.double)
a[1:4, 1:4] = inner_square

b = np.full((5,5), 50, np.double)
b[1:4, 1:4] = inner_square

c = np.full((5,5), 100, np.double)
c[1:4, 1:4] = inner_square

d = np.full((5,5), 150, np.double)
d[1:4, 1:4] = inner_square

simultaneous=np.hstack((a,b,c,d))


im=plt.imshow(simultaneous, cmap='gray',interpolation='nearest',vmin=0, vmax=255) 
#plt.rcParams["figure.figsize"] = (70,10)
plt.axis('off')
plt.colorbar(im, orientation='horizontal')
plt.show()

e = np.full((9,5), 200, np.double)
f = np.full((9,5), 150, np.double)
g = np.full((9,5), 100, np.double)
h = np.full((9,5), 75, np.double)
i = np.full((9,5), 50, np.double)
image1= np.hstack((e,f,g,h,i))

e[:,4] = 255
f[:,4] = 255
g[:,4] = 255
h[:,4] = 255
i[:,4] = 255
image2=np.hstack((e,f,g,h,i))

plt.subplot(1,2,1)
plt.imshow(image1, cmap='gray',vmin=0, vmax=255,interpolation='nearest',aspect=4) 
plt.title('Bands')
plt.axis('off')


plt.subplot(1,2,2)
plt.imshow(image2, cmap='gray',vmin=0, vmax=255,interpolation='nearest',aspect=4) 
plt.title('Bands with white breaks')
plt.axis('off')

plt.show()

strips = np.linspace( 0, 255, 10, np.double)  
strips = strips.reshape((-1, 1))
M = np.linspace( 255, 0, 10, np.double)   
n = np.ones((20, 10), np.double)

background = n[:,:]*M
background[5:15,::2] = strips

without_background = np.full((20,10), 255, np.double)
without_background[5:15,::2] = strips

plt.subplot(1,2,1)
plt.imshow(background, cmap='gray',vmin=0, vmax=255,interpolation='nearest') 
plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')


plt.subplot(1,2,2)
plt.imshow(without_background, cmap='gray',vmin=0, vmax=255,interpolation='nearest')
plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')


plt.show()



