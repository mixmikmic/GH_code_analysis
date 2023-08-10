from __future__ import division
get_ipython().magic('pylab inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from scipy import misc
xsharp = misc.ascent()
print(xsharp.shape) # like Matlab's size(xsharp). Given as a tuple.
print("The size of the image is %s x %s." % (xsharp.shape[0],xsharp.shape[1]))
print("The range of the pixel values is [%s,%s]." % (xsharp.min(),xsharp.max()))
xsharp = xsharp.astype(float32) 

figsize(11,11)
imshow(xsharp, interpolation='nearest', cmap=cm.gray, vmin=0, vmax=255)
plt.axis("off")
# Without specifying vmin and vmax, imshow auto-adjusts its range so that black and white are
# the min and max of the data, respectively, like Matlab's imagesc.
colorbar()       # displays the color bar close to the image
#axis('off')     # uncomment to remove the axes
subplots_adjust(top=0.75)
title('This is Ascent')

mask = rand(xsharp.shape[0],xsharp.shape[1])>0.80

fig, (subfig1,subfig2) = subplots(1,2,figsize=(16,7)) # one figure with two horizontal subfigures
subfig1.imshow(mask, cmap=cm.gray)
subfig2.imshow(mask*xsharp, cmap=cm.gray)
subfig1.set_title('The binary mask')
subfig2.set_title('The available pixel values are displayed, the missing pixels are in black')

y = mask*xsharp

D = lambda x : c_['2,3',r_[diff(x,1,0), zeros([1,x.shape[1]])],c_[diff(x,1,1), zeros([x.shape[0],1])]]

Dadj = lambda v : r_['0,2',-v[0,:,0],-diff(v[:-1,:,0],1,0),v[-2,:,0]] + c_['1,2',-v[:,0,1],-diff(v[:,:-1,1],1,1),v[:,-2,1]]

def prox_g_conj (u, Lambda) :
    return u/tile(maximum(sqrt(sum(u**2,2,keepdims=True))/Lambda,1),(1,1,2)) # soft-thresholding

A = mask

def prox_f (x, y) :
    x[mask]=y[mask]
    return x

tau = 1
rho = 1
sigma = 1/tau
nbiter = 1500

(N1,N2) = shape(xsharp)
x = zeros([N1,N2])
u = zeros([N1,N2, 2])
En_array = zeros(nbiter)
for iter in range(nbiter):  # iter goes from 0 to nbiter-1
    xtilde = prox_f(x - tau*Dadj(u),y)
    utilde = prox_g_conj(u + sigma*D(2*xtilde-x), sigma)
    x = x + rho*(xtilde - x)
    u = u + rho*(utilde - u)
    En_array[iter] = sum(sqrt(sum(D(x)**2,2)))
    
x_restored = x

figsize(9,9)
imshow(x_restored, interpolation='nearest', cmap=cm.gray, vmin=0, vmax=255)
title('Denoised image')
plt.axis("off")

figsize(15,7)
plot(log10(En_array))

#math and linear algebra stuff
import numpy as np
from skimage import measure
from scipy import misc
import scipy.signal as scis
import matplotlib.pyplot as plt

"""
  import dataset, and convert to fp64
"""
xbar = misc.ascent().astype(np.float64)
dyn = xbar.ptp()
#Plot
plt.figure(0,figsize=(10,10))
plt.title("Original image")
plt.imshow(xbar,interpolation="nearest", cmap=plt.cm.gray)
plt.axis("off")

"""
  Define a 2D, non axis aligned gaussian kernel, and its time reversed version
"""
#define mesh
sizeK = 11 # must be odd
X,Y=np.meshgrid(np.linspace(-1,1,sizeK),np.linspace(-1,1,sizeK))
xvec = np.array([np.reshape(X,X.size),np.reshape(Y,Y.size)])
#define a 2D rotation matrix:
def GetRotMat( theta ):
  return np.array([[np.cos(theta),-np.sin(theta)],
    [np.sin(theta),np.cos(theta)]])
Mrot = GetRotMat( 3.*np.pi/4. )
#define anisotropic and rotated gaussian
Sigma = np.dot(Mrot,np.dot(np.diag([0.15,1]),Mrot.T))
mu = np.array([[0],[0]])
coef = 1/np.sqrt(np.power(2*np.pi,2)*np.linalg.det(Sigma))
#2-dimensional gaussian pdf
test=np.dot(np.linalg.inv(Sigma),xvec-mu)
K=coef * np.exp(-0.5* np.sum(test*(xvec-mu),0))
K=np.reshape(K,X.shape)/K.sum()
#Normalize
K/=K.sum()
KT=K[::-1,::-1]
#Plot
plt.figure(1,figsize=(5,5))
plt.title("2D gaussian kernel")
plt.imshow(K)
plt.axis("off")

"""
  Perform convolution with gaussian kernel + add noise
"""
blurred=scis.convolve2d(xbar, K, mode='same', boundary='symm', fillvalue=0)
stddev = 10
blurred += np.random.normal(0,stddev,blurred.shape)
psnrBlurred = measure.compare_psnr(xbar, blurred, dynamic_range=dyn)
ssimBlurred = measure.compare_ssim(xbar, blurred, win_size=5, dynamic_range=dyn)
#Plot
plt.figure(2,figsize=(10,10))
plt.title("Blurred image, PSNR= "+("%.2f" % psnrBlurred)+
  " SSIM= "+("%.2f" % ssimBlurred))
plt.imshow(blurred, interpolation="nearest", cmap=plt.cm.gray)
plt.axis("off")

#A can be an arbitrary linear operator
A = lambda x : scis.convolve2d(x, K, mode='same', boundary='symm', fillvalue=0)
AT = lambda x : scis.convolve2d(x, KT, mode='same', boundary='symm', fillvalue=0)

def GetLNormThroughPowerMethod(init):
  """
    Perform a few iteration of the power method in order to obtain
    the maximum eigenvalue of the L^*L operator
  """
  x=init.copy()
  for i in range(10):
    x = AT(A(x))+Dadj(D(x))
    s = np.linalg.norm(x)
    x /= s
  return np.sqrt(s)

def prox_f(inf) :
  """

  """
  return inf

def prox_g_conj_1 (p, y, sigma) :
  """

  """
  return (p-sigma*y)/(0.5*sigma+1.)

def prox_g_conj_2 (u, Lambda) :
  """
    Proximity operator for the G_2* function
  """
  ret = u.copy()
  n = np.maximum(np.sqrt(np.sum(u**2, 2))/Lambda, 1.0)
  ret[:,:,0]/=n
  ret[:,:,1]/=n
  return ret

Lnorm = GetLNormThroughPowerMethod(AT(blurred))*1.1 #take 10% margin
tau = 1./Lnorm
sigma = 1./Lnorm
rho = 1. #rho > 1 allows to speed up through momentum effect
nbIter = 750
lambdaTV = 1.

xk = np.zeros_like(blurred)  #primal var at current iteration
x_tilde = np.zeros_like(xk)  #primal var estimator
p = np.zeros_like(blurred) #dual var 1
q = np.zeros([xbar.shape[0],xbar.shape[1],2]) #dual var 2
primObj = np.zeros(nbIter)
dualObj = np.zeros_like(primObj)
for iter in range(nbIter):  # iter goes from 0 to nbIter-1
  p = prox_g_conj_1( p + sigma * A(x_tilde), blurred, sigma )
  q = prox_g_conj_2( q + sigma * D(x_tilde),lambdaTV )
  dualObj[iter] = -0.25*np.linalg.norm(p)**2 - np.dot(p.flatten(),blurred.flatten())
  xk1 = xk
  xk = prox_f( xk - tau * Dadj(q) - tau * AT(p) )
  x_tilde = xk + rho*( xk - xk1 )
  primObj[iter] = lambdaTV * np.sum(np.sqrt(np.sum(D(xk)**2,2))) +     np.linalg.norm(A(xk)-blurred)**2

#Evaluate quality
psnrXk = measure.compare_psnr(xbar, xk, dynamic_range=dyn)
ssimXk = measure.compare_ssim(xbar, xk, win_size=5, dynamic_range=dyn)

#Plot
plt.figure(3,figsize=(10,10))
plt.title("Reconstructed image, PSNR= "+("%.2f" % psnrXk)+
  " SSIM= "+("%.2f" % ssimXk))
plt.imshow(xk, interpolation="nearest", cmap=plt.cm.gray)
plt.axis("off")
plt.figure(4,figsize=(10,10))
plt.xlabel("Iteration index")
plt.ylabel("Primal and dual objective value (logscale)")
plt.plot(range(nbIter),np.log10(primObj),label="Primal objective")
plt.plot(range(nbIter),np.log10(dualObj),label="Dual objective")
plt.title("Value of the composite objective along the iterations")
plt.legend()
plt.show()
plt.figure(5,figsize=(10,10))
plt.xlabel("Iteration index")
plt.ylabel("Primal-dual gap (logscale)")
plt.plot(range(nbIter),np.log10(np.abs(primObj-dualObj)))
plt.title("Primal-dual gap along the iterations")



