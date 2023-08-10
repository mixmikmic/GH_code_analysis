import numpy as np
import matplotlib.pyplot as plt
import rect as r
import prop
L1 = 0.5 #side length in m
M = 250 #number of samples
step1 = L1/M #step size in m
x1 = np.linspace(-L1/2,L1/2,M) #input support coordinates
y1 = x1
energy = 5 # in keV
wavel = 0.5*10**(-6) # wavelength of light in m
k = 2*np.pi/wavel #wavevector
w = 0.051 #width of square aperture
z = 1000 # propogation distance in meters

X1,Y1 = np.meshgrid(x1,y1)
u1 = np.multiply(r.rect(X1/(2*w)),r.rect(Y1/(2*w))) #creating the input beam profile
I1 = abs(np.multiply(u1,u1))  #input intensity profile

plt.figure()
plt.suptitle('Input beam')
plt.imshow(np.abs(I1),extent=[x1[0],x1[-1],y1[0],y1[-1]])
plt.show()

z = 1000 # propogation distance

ua = prop.propTF(u1,step1,L1,wavel,z) #result using TF method
Ia = abs(ua) ** 2 #TF output intensity profile

ub = prop.propIR(u1,step1,L1,wavel,z) #result using IR method
Ib = abs(ub) ** 2 #IR output intensity profile
'''
Plotting.
'''

plt.figure()
plt.suptitle('propogation distance (in meters) = '+str(z))
plt.subplot(121)
plt.imshow(abs(Ia),extent=[x1[0],x1[-1],y1[0],y1[-1]])
plt.title('TF')
plt.subplot(122)
plt.imshow(abs(Ib),extent=[x1[0],x1[-1],y1[0],y1[-1]])
plt.title('IR')
plt.show()

z = 2000 # propogation distance
ua = prop.propTF(u1,step1,L1,wavel,z) #result using TF method
Ia = np.abs(ua) ** 2 #TF output intensity profile

ub = prop.propIR(u1,step1,L1,wavel,z) #result using IR method
Ib = np.abs(ub) ** 2 #IR output intensity profile

'''
Plotting.
'''

plt.figure()
plt.suptitle('propogation distance (in meters) = '+str(z))
plt.subplot(121)
plt.imshow(np.abs(Ia),extent=[x1[0],x1[-1],y1[0],y1[-1]])
plt.title('TF')
plt.subplot(122)
plt.imshow(np.abs(Ib),extent=[x1[0],x1[-1],y1[0],y1[-1]])
plt.title('IR')
plt.show()

z = 4000 # propogation distance

ua = prop.propTF(u1,step1,L1,wavel,z) #result using TF method
Ia = np.abs(ua) ** 2 #TF output intensity profile

ub = prop.propIR(u1,step1,L1,wavel,z) #result using IR method
Ib = np.abs(ub) ** 2 #IR output intensity profile

'''
Plotting.
'''

plt.figure()
plt.suptitle('propogation distance (in meters) = '+str(z))
plt.subplot(121)
plt.imshow(abs(Ia),extent=[x1[0],x1[-1],y1[0],y1[-1]])
plt.title('TF')
plt.subplot(122)
plt.imshow(abs(Ib),extent=[x1[0],x1[-1],y1[0],y1[-1]])
plt.title('IR')
plt.show()

z = 20000 # propogation distance

ua = prop.propTF(u1,step1,L1,wavel,z) #result using TF method
Ia = np.abs(ua) ** 2 #TF output intensity profile

ub = prop.propIR(u1,step1,L1,wavel,z) #result using IR method
Ib = np.abs(ub) ** 2 #IR output intensity profile

'''
Plotting.
'''

plt.figure()
plt.suptitle('propogation distance = '+str(z))
plt.subplot(121)
plt.imshow(abs(Ia),extent=[x1[0],x1[-1],y1[0],y1[-1]])
plt.title('TF')
plt.subplot(122)
plt.imshow(abs(Ib),extent=[x1[0],x1[-1],y1[0],y1[-1]])
plt.title('IR')
plt.show()

from IPython.display import Image
Image(filename='table_5_1.png') 

z = 10000 # propogation distance

ua = prop.prop1FT(u1,step1,L1,wavel,z) #result using TF method
Ia, L2 = np.abs(ua) ** 2 #TF output intensity profile

'''
Plotting.
'''

plt.figure()
plt.suptitle('propogation distance = '+str(z))
plt.subplot(111)
plt.imshow(Ia,extent=[x1[0],x1[-1],y1[0],y1[-1]])
plt.show()

u1 = np.load('prop_demos/exiting.npy')
I1 = np.abs(u1) ** 2
plt.figure()
plt.suptitle('Wavefront before single FT propagation')
plt.imshow(np.abs(I1))
plt.show()
delta = 1
kev = 5
lmbda_nm = 1.24 / kev
N = u1.shape[0]

z_eq_samp = delta ** 2 * N / lmbda_nm
print z_eq_samp

z = 5000 # propogation distance

ua = prop.prop1FT(u1, delta, delta * N, lmbda_nm, z) #result using TF method
Ia, L2 = np.abs(ua) ** 2 # output intensity profile

'''
Plotting.
'''

plt.figure()
plt.suptitle('propogation distance = '+str(z))
plt.subplot(111)
plt.imshow(Ia)
plt.show()

udc = u1[0, 0]
u1 = u1 - u1[0, 0]
ua = prop.prop1FT(u1, delta, delta * N, lmbda_nm, z)
u_max = v_max = 1 / (2 * delta)
k = 2 * np.pi / lmbda_nm
udc *= np.exp(1j * k * z * np.sqrt(1 - lmbda_nm ** 2 * (u_max**2 - v_max**2)))
ua += udc
Ia, L2 = np.abs(ua) ** 2 # output intensity profile

plt.figure()
plt.suptitle('propogation distance = '+str(z))
plt.subplot(111)
plt.imshow(Ia)
plt.show()

w = 0.011 #width of square aperture
X1,Y1 = np.meshgrid(x1,y1)
u1 = np.multiply(r.rect(X1/(2*w)),r.rect(Y1/(2*w))) #creating the input beam profile
I1 = abs(np.multiply(u1,u1))  #input intensity profile
plt.figure()
plt.suptitle('Input beam')
plt.imshow(np.abs(I1),extent=[x1[0],x1[-1],y1[0],y1[-1]])
plt.show()

z = 2000 # propogation distance

u2,L2 = prop.propFF(u1,step1,L1,wavel,z) #result using TF method
I2 = np.abs(u2) ** 2 #TF output intensity profile

'''
output grid
'''
x2 = np.linspace(-L2/2.,L2/2.,M) #input support coordinates
y2 = x2
X2,Y2 = np.meshgrid(x2,y2)


'''
Plotting.
'''

plt.figure()
plt.suptitle('propogation distance = '+str(z))
plt.imshow(abs(I2)**(1./3),extent=[x2[0],x2[-1],y2[0],y2[-1]])
plt.show()

plt.figure()
plt.plot(x2,I2[126,:])
plt.title('Output beam intensity profile')
plt.show()





