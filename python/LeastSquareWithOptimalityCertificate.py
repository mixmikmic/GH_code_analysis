#math and linear algebra stuff
import numpy as np
from skimage import measure
from scipy import misc
import scipy.signal as scis
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

"""
  import dataset, and convert to fp32
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

#Plot it
plt.figure(1,figsize=(5,5))
plt.title("2D gaussian kernel")
plt.imshow(K)
plt.axis("off")

"""
 Check that AT and T are adjoint of each other
"""
#A can be an arbitrary linear operator, here it is the blurring operator
A = lambda x : scis.convolve2d(x, K, mode='same', boundary='symm', fillvalue=0)
AT = lambda x : scis.convolve2d(x, KT, mode='same', boundary='symm', fillvalue=0)

a = np.random.uniform(1,4,xbar.shape)
b = np.random.uniform(1,4,xbar.shape)

#check that <ATa,b> = <a,Ab>
ATa = np.dot(AT(a).flatten(),b.flatten())
aAb = np.dot(a.flatten(),A(b).flatten())
assert( np.isclose(ATa,aAb) )

"""
  Perform convolution with gaussian kernel
"""
blurred=A(xbar)
stddev = 10
psnrBlurred = measure.compare_psnr(xbar, blurred, dynamic_range=dyn)
ssimBlurred = measure.compare_ssim(xbar, blurred, win_size=5, dynamic_range=dyn)

#Plot
plt.figure(2,figsize=(10,10))
plt.title("Blurred image, PSNR= "+("%.2f" % psnrBlurred)+
  " SSIM= "+("%.2f" % ssimBlurred))
plt.imshow(blurred, interpolation="nearest", cmap=plt.cm.gray)
plt.axis("off")

def GetLNormThroughPowerMethod(init,nbIter):
  """
    Perform a few iteration of the power method in order to obtain
    the maximum eigenvalue of the L^*L operator
  """
  x=init.copy()
  for i in range(nbIter):
    x = AT(A(x))
    s = np.linalg.norm(x)
    x /= s
  return np.sqrt(s)

def prox_f(x) :
  """
  """
  return x

def prox_g_conj (u, y, gamma) :
  """
  """
  return (u-gamma*y)/(gamma+1.)

Lnorm = GetLNormThroughPowerMethod(AT(blurred),10)*1.1 #take 10% margin
tau = 1./Lnorm
sigma = 1./Lnorm
rho = 1. #rho > 1 allows to speed up through momentum effect
nbIter = 200

xk = np.zeros_like(AT(blurred))  #primal var at current iteration
x_tilde = np.zeros_like(xk)  #primal var estimator
uk = np.zeros_like(blurred) #Dual variable
primObj = np.zeros(nbIter)
dualObj = np.zeros_like(primObj)
for iter in range(nbIter):  # iter goes from 0 to nbIter-1
  uk = prox_g_conj( uk + sigma * A(x_tilde), blurred, sigma )
  dualObj[iter] = -0.5*np.linalg.norm(uk)**2 - np.dot(uk.flatten(),blurred.flatten())
  xk1 = xk
  xk = prox_f( xk - tau * AT(uk) )
  x_tilde = xk + rho*( xk - xk1 )
  primObj[iter] = np.linalg.norm(A(xk)-blurred)**2

#Evaluate quality
psnrXk = measure.compare_psnr(xbar, xk, dynamic_range=dyn)
ssimXk = measure.compare_ssim(xbar, xk, win_size=5, dynamic_range=dyn)

#Show reconstruction
plt.figure(3,figsize=(10,10))
plt.title("Reconstructed image, PSNR= "+("%.2f" % psnrXk)+
  " SSIM= "+("%.2f" % ssimXk))
plt.imshow(xk, interpolation="nearest", cmap=plt.cm.gray)
plt.axis("off")

#Show convergence of primal/dual
plt.figure(4)
plt.xlabel("Iteration index")
plt.ylabel("Primal and dual objective value (logscale)")
plt.plot(range(nbIter),primObj,label="Primal objective")
plt.plot(range(nbIter),dualObj,label="Dual objective")
plt.title("Value of the composite objective along the iterations")
plt.legend()

#Show that PD gap is numerically close to 0
plt.figure(5)
plt.xlabel("Iteration index")
plt.ylabel("Primal-dual gap (logscale)")
plt.plot(range(nbIter),np.log10(np.abs(primObj-dualObj)))
plt.title("Primal-dual gap along the iterations")
plt.show()



