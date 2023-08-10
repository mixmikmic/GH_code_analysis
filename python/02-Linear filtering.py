get_ipython().magic('matplotlib inline')
import sys
sys.path.insert(0,'..')
from IPython.display import HTML,Image,SVG,YouTubeVideo
from helpers import header

HTML(header())

Image('http://homepages.ulb.ac.be/~odebeir/data/b.png')

Image('http://homepages.ulb.ac.be/~odebeir/data/b1.png')

Image('http://homepages.ulb.ac.be/~odebeir/data/b2.png')

Image('http://homepages.ulb.ac.be/~odebeir/data/b3.png')

from skimage.morphology import disk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.data import camera,lena
plt.style.use('ggplot')


Image('http://homepages.ulb.ac.be/~odebeir/data/b5.png')

Image('http://homepages.ulb.ac.be/~odebeir/data/b6.png')

Image('http://homepages.ulb.ac.be/~odebeir/data/conv1.png')

import bokeh.plotting as bk
from helpers import bk_image,bk_image_hoover,bk_compare_image
bk.output_notebook()

from skimage.morphology import disk
from skimage.data import camera
from skimage.filters.rank import mean

g = camera()[-1:2:-1,::2]
f = mean(g,disk(1))

bk_compare_image(g,f)

f = mean(g,disk(4))
bk_compare_image(g,f)

import numpy as np
from numpy.fft import fft,fftshift,ifftshift
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def step_power(length):
    nsample = 100
    t = np.arange(nsample)
    x = t<=length
    fx = fftshift(fft(x))
    power = np.abs(fx)**2 
    angle = np.angle(fx)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.gca().set_ylim([0,1.2])
    plt.title('$f(x)$')
    plt.plot(x)
    plt.subplot(1,2,2)
    plt.plot(np.arange(-nsample/2,nsample/2),power);
    plt.title('$|F(u)|^2$')
    
step_power(1)
step_power(4)
step_power(50)

def sine_power(freq):
    nsample = 100
    t = np.linspace(-np.pi,np.pi,nsample)
    x = np.sin(t*freq)
    fx = fftshift(fft(x))
    power = np.abs(fx)**2 
    angle = np.angle(fx)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.gca().set_ylim([-1.1,1.1])
    plt.title('$f(x)$')
    plt.plot(x)
    plt.subplot(1,2,2)
    plt.plot(np.arange(-nsample/2,nsample/2),power);
    plt.title('$|F(u)|^2$')
    
sine_power(1)
sine_power(4)
sine_power(10)

def mix_sine_power(freq):
    nsample = 100
    t = np.linspace(-np.pi,np.pi,nsample)
    x = np.zeros_like(t)
    for f,a in freq:
        x += np.sin(t*f)*a
    fx = fftshift(fft(x))
    power = np.abs(fx)**2 
    angle = np.angle(fx)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.gca().set_ylim([-1.1,1.1])
    plt.plot(x)
    plt.title('$f(x)$')
    plt.subplot(1,2,2)
    plt.plot(np.arange(-nsample/2,nsample/2),power);
    plt.title('$|F(u)|^2$')
    
mix_sine_power([(1,.5),(4,.2),(10,.3)])
mix_sine_power([(1,.5),(4,.3),(10,.2)])

import matplotlib.pyplot as plt
import matplotlib.pylab as plab
import matplotlib.cm as cm
import scipy as scp
import numpy as np
from numpy.fft import fft2,ifft2,fftshift,ifftshift
from scipy import ndimage
from mpl_toolkits.mplot3d import axes3d

w = 2
n = 128
r = np.zeros((n,n),dtype = np.complexfloating)
r[n/2-w:n/2+w,n/2-w:n/2+w] = 1.0 + 0.0j

F = fft2(r) #2D FFT of the image
rr = ifft2(F) #2D inverse FFT of F 

real_part = F.real
imag_part = F.imag

fig = plt.figure(1)
plt.imshow(r.real,interpolation='nearest',origin='lower',cmap=cm.gray)
plt.colorbar()
plt.title('$f(x,y)$');

fig = plt.figure(2)
plt.subplot(1,2,1)
plt.imshow(fftshift(real_part),interpolation='nearest',origin='lower')
plt.title('$\mathcal{R} \{ F(u,v)\}$')

plt.subplot(1,2,2)
plt.imshow(fftshift(imag_part),interpolation='nearest',origin='lower')

plt.title('$\mathcal{I} \{ F(u,v)\}$')

fig = plt.figure(3)
plt.imshow(np.abs(fftshift(F))**2,interpolation='nearest',origin='lower')

plt.title('$|F(u,v)|^2$');

fig = plt.figure(figsize=[10,10])
ax = fig.gca(projection='3d')
X,Y = np.meshgrid(range(n),range(n))
Z = np.abs(fftshift(F))**2
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='x', offset=-10, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='y', offset=140, cmap=cm.coolwarm)

plt.xlabel('$u$')
plt.ylabel('$v$')
plt.title('$|F(u,v)|^2$');

fig = plt.figure()
plt.imshow(rr.real,interpolation='nearest',origin='lower',cmap=cm.gray)
plt.colorbar()
plt.title('$\mathcal{F}^{-1}(F(u,v))$');

ima = camera()[-1::-1,:]
F = fft2(ima)
amplitude = np.abs(F)
angle = np.angle(F)

noise = np.random.random(ima.shape)
n_amplitude = amplitude * (1+10*noise)

real = n_amplitude * np.cos(angle)
imag = n_amplitude * np.sin(angle)

n_F = real + 1j *imag

n_ima = np.real(ifft2(n_F))

fig = plt.figure(figsize=[10,5])
plt.subplot(1,2,1)
plt.imshow(ima,interpolation='nearest',origin='lower')
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(n_ima,interpolation='nearest',origin='lower')
plt.colorbar();
plt.title('amplitude * (1+10*noise)');

ima = camera()[-1::-1,:]
F = fft2(ima)
amplitude = np.abs(F)
angle = np.angle(F)

noise = np.random.random(ima.shape)
n_angle = angle * (1+.5*noise)


real = amplitude * np.cos(n_angle)
imag = amplitude * np.sin(n_angle)

n_F = real + 1j *imag

n_ima = np.real(ifft2(n_F))

fig = plt.figure(figsize=[10,5])
plt.subplot(1,2,1)
plt.imshow(ima,interpolation='nearest',origin='lower')
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(n_ima,interpolation='nearest',origin='lower')
plt.colorbar();
plt.title('angle * (1+.5*noise)');

from skimage.data import camera
from skimage.filters import rank as skr
from skimage.morphology import disk

im = camera()[-1::-1,::]

w,h = im.shape

n = 32

#square filter
s = np.zeros(im.shape,dtype = np.complexfloating)
s[w/2-n:w/2+n,h/2-n:h/2+n] = 1.0 + 0.0j

#circular filter 
c = np.zeros(im.shape,dtype = np.complexfloating)
for i in range(w):
    for j in range(h):
        if ((i-w/2)**2 + (j-h/2)**2)<(n*n):
            c[i,j] = 1.0 + 0.0j
            
#smooth filter borders
c = skr.mean(np.real(c*255).astype('uint8'),disk(10))
c = c.astype(np.complexfloating)/(255.0+0j)

F1 = fft2(im.astype(np.complexfloating))
F3 = F1*ifftshift(s)
F4 = F1*ifftshift(c)

#high pass using the complement of c
F5 = F1*ifftshift((1.0 + 0j)-c)

psF1 = (F1**2).real

low_pass_rec = ifft2(F3) 
low_pass_circ = ifft2(F4)
high_pass_circ = ifft2(F5) 

fig = plt.figure(1)
plt.imshow(np.abs(fftshift(psF1))**2,interpolation='nearest',origin='lower',vmax = 10e18)
plt.title('$|F(u,v)|^2$')
plt.colorbar()

fig = plt.figure(2)
plt.subplot(1,2,1)
plt.imshow(s.real,interpolation='nearest',origin='lower')
plt.title('$|G(u,v)|$')
plt.subplot(1,2,2)
plt.imshow(low_pass_rec.real,interpolation='nearest',origin='lower',cmap=cm.gray)
plt.title('$g(x,y)*f(x,y)$');

fig = plt.figure(3)
plt.subplot(1,2,1)
plt.imshow(c.real,interpolation='nearest',origin='lower')
plt.title('$g(x,y)$')
plt.subplot(1,2,2)
plt.imshow(low_pass_circ.real,interpolation='nearest',origin='lower',cmap=cm.gray)
plt.title('$g(x,y)*f(x,y)$');

fig = plt.figure(4)
plt.subplot(1,2,1)
plt.imshow(1.0-c.real,interpolation='nearest',origin='lower')
plt.title('$g(x,y)')
plt.subplot(1,2,2)
plt.imshow(high_pass_circ.real,interpolation='nearest',origin='lower',cmap=cm.gray)
plt.title('$g(x,y)*f(x,y)$');

from skimage.data import camera


im = camera().astype(np.complexfloating)
target_center = (314,375)
w = 16
crop = im[target_center[1]-w:target_center[1]+w,target_center[0]-w:target_center[0]+w]
m,n = im.shape

g = np.zeros((m,n),dtype = np.complexfloating)
g[n/2-w:n/2+w,m/2-w:m/2+w] = crop - np.mean(im) + .0j

#normalize
f = im-np.mean(im)


plt.figure(2)
plt.imshow(g.real,interpolation='nearest',cmap=cm.gray)
plt.colorbar()
plt.title('$g(x,y)$');

F = fft2(f)
G = fft2(np.conjugate(g))

R = F*G
r = fftshift(np.real(ifft2(R)))

corr = 255*((r-np.min(r))/(np.max(r)-np.min(r)))
#corr = 255*(255*((r-np.min(r))/(np.max(r)-np.min(r)))>250)

bk_compare_image(np.real(im[-1::-1,:]),np.real(corr[-1::-1,:]))

from skimage.io import imread
im = imread('http://homepages.ulb.ac.be/~odebeir/data/moire1.png').astype(np.float)
f = fft2(im)

power = fftshift(np.abs(f)**2)
pmax = np.max(power)

plt.figure(figsize=[10,5])
plt.subplot(1,2,1)
plt.imshow(im,cmap = cm.gray)
plt.title('$f(x,y)$');
plt.subplot(1,2,2)
plt.imshow(power,vmin=0,vmax=pmax/5000.);
plt.title('$|F(u,v)|^2$');



