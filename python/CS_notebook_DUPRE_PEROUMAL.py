get_ipython().magic('pylab inline')
import matplotlib.pyplot as plt
import numpy as np
import pyfits as pyf
#import algorithms and the starlet transform
import models

# Get the data

hdulist = pyf.open('ngc2997.fits')
x = np.double(hdulist[0].data)
hdulist.close()

plt.imshow(x)
plt.title('Ground truth')
plt.show()

# Get sampled measurements

for i in range(len(x)):
    for j in range(len(x[i])):
        if random.uniform(0, 1) > 0.7:
            x[i][j] = 0

plt.imshow(x)
plt.show()

# Add some noise

n = 50 * np.random.randn(256, 256)
bn = x + n

plt.imshow(bn)
plt.title('Incoherent sampled measurements')
plt.show()

# Compute the reconstructed image

x_st = models.SoftTHRD(bn, J=3, kmad=40)
print('SoftTHRD done')
x_ht = models.HardTHRD(bn, J=3, kmad=40)
print('HardTHRD done')
x_mr, mask = models.MRDenoise(bn, J=3, nmax=25, kmad=40)
print('MRDenoise done')

plt.imshow(x_ht)
plt.title('Hard-Thresholding')
plt.show()

plt.imshow(x_st)
plt.title('Soft-Thresholding')
plt.show()

plt.imshow(x_mr)
plt.title('MR Denoise')
plt.show()

err_bn = []
err_st = []
err_ht = []
err_mr = []

# Get the data

hdulist = pyf.open('data_tp2/ngc2997.fits')
x = np.double(hdulist[0].data)
hdulist.close()

for i in range(8):

    n = i * 50 * np.random.randn(256, 256)

    bn = x + n
    err_bn.append(
        np.linalg.norm(x - bn, ord=None, axis=None, keepdims=False) /
        np.linalg.norm(x, ord=None, axis=None, keepdims=False))

    x_st = models.SoftTHRD(bn, J=3, kmad=2.4)
    err_st.append(
        np.linalg.norm(x - x_st, ord=None, axis=None, keepdims=False) /
        np.linalg.norm(x, ord=None, axis=None, keepdims=False))
    x_ht = models.HardTHRD(bn, J=3, kmad=2.4)
    err_ht.append(
        np.linalg.norm(x - x_ht, ord=None, axis=None, keepdims=False) /
        np.linalg.norm(x, ord=None, axis=None, keepdims=False))
    x_mr, mask = models.MRDenoise(bn, J=3, nmax=25, kmad=2.4)
    err_mr.append(
        np.linalg.norm(x - x_mr, ord=None, axis=None, keepdims=False) /
        np.linalg.norm(x, ord=None, axis=None, keepdims=False))

    print(i + 1, 'iterations done')

plt.figure(figsize=(20, 10))
a = plt.scatter(range(0, 351, 50), err_bn, color='blue', marker='x', s=100)
b = plt.scatter(range(0, 351, 50), err_ht, color='red', marker='o', s=100)
c = plt.scatter(range(0, 351, 50), err_st, color='black', marker='v', s=100)
d = plt.scatter(range(0, 351, 50), err_mr, color='cyan', marker='^', s=100)
plt.legend(
    (a, b, c, d), ('Noisy signal', 'Hard-thresholding', 'Soft-thresholding',
                   'MRDenoise'),
    loc='upper left')
plt.xlabel('Noise coefficient')
plt.ylabel('Error')
plt.show()

from copy import deepcopy as dp
err_bn = []
err_st = []
err_ht = []
err_mr = []

for k in range(10):

    hdulist = pyf.open('data_tp2/ngc2997.fits')
    x = np.double(hdulist[0].data)
    hdulist.close()
    
    x2 = dp(x)

    # Get sampled measurements

    for i in range(len(x)):
        for j in range(len(x[i])):
            if random.uniform(0, 1) < (k+1)*0.1:
                x2[i][j] = 0

    n = 0 * np.random.randn(256, 256)

    bn = x2 + n

    err_bn.append(
        np.linalg.norm(x - bn, ord=None, axis=None, keepdims=False) /
        np.linalg.norm(x, ord=None, axis=None, keepdims=False))

    x_st = models.SoftTHRD(bn, J=3, kmad=2.4)
    err_st.append(
        np.linalg.norm(x - x_st, ord=None, axis=None, keepdims=False) /
        np.linalg.norm(x, ord=None, axis=None, keepdims=False))
    x_ht = models.HardTHRD(bn, J=3, kmad=2.4)
    err_ht.append(
        np.linalg.norm(x - x_ht, ord=None, axis=None, keepdims=False) /
        np.linalg.norm(x, ord=None, axis=None, keepdims=False))
    x_mr, mask = models.MRDenoise(bn, J=3, nmax=25, kmad=2.4)
    err_mr.append(
        np.linalg.norm(x - x_mr, ord=None, axis=None, keepdims=False) /
        np.linalg.norm(x, ord=None, axis=None, keepdims=False))

    print(k + 1, 'iterations done')

plt.figure(figsize=(20, 10))
a = plt.scatter(range(10, 101, 10), err_bn, color='blue', marker='x', s=100)
b = plt.scatter(range(10, 101, 10), err_ht, color='red', marker='o', s=100)
c = plt.scatter(range(10, 101, 10), err_st, color='black', marker='v', s=100)
d = plt.scatter(range(10, 101, 10), err_mr, color='cyan', marker='^', s=100)
plt.legend(
    (a, b, c, d), ('Sampled signal', 'Hard-thresholding', 'Soft-thresholding',
                   'MRDenoise'),
    loc='upper left')
plt.xlabel('Sparsity (percentage)')
plt.ylabel('Error')
plt.show()


# Get the data

hdulist = pyf.open('AstroImages.fits')
x=[]
x.append(np.double(hdulist[0].data[0]))
x.append(np.double(hdulist[0].data[1]))
x.append(np.double(hdulist[0].data[2]))
hdulist.close()

fig=plt.figure(figsize=(20,10))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(x[i])
    plt.title('Ground truth image No'+str(i+1))
plt.show()

# Add some noise
bn=[0,0,0]
for i in range(3):
    n = 0.002 * np.random.randn(256, 256)
    bn[i] = x[i] + n
    
fig=plt.figure(figsize=(20,10))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(bn[i])
    plt.title('Noisy image No'+str(i+1))
plt.show()

# Compute the reconstructed image
x_st=[0,0,0]
x_ht=[0,0,0]
x_mr=[0,0,0]
for i in range(3):
    
    x_st[i] = models.SoftTHRD(bn[i], J=3, kmad=40)
    x_ht[i] = models.HardTHRD(bn[i], J=3, kmad=40)
    x_mr[i], mask = models.MRDenoise(bn[i], J=3, nmax=25, kmad=40)
    print('Image ',i+1,' processed')

fig=plt.figure(figsize=(20,10))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(x_st[i])
    plt.title('Soft-thresholding image No'+str(i+1))
plt.show()

fig=plt.figure(figsize=(20,10))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(x_ht[i])
    plt.title('Hard-thresholding image No'+str(i+1))
plt.show()

fig=plt.figure(figsize=(20,10))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(x_mr[i])
    plt.title('MR Denoise image No'+str(i+1))
plt.show()



