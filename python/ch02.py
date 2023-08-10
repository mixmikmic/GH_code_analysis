import numpy as np
import math
import matplotlib.pyplot as plt

plt.style.use('ggplot')
get_ipython().magic('matplotlib inline')

# axes
Dd = 0.1
N = 101
d = Dd * np.arange(N)
dmin = 0
dmax = 10

# build a normal distribution
dbar = 5
sigma = 1
sigma2 = sigma ** 2
p = np.exp(-0.5 * np.multiply((d-dbar), (d-dbar)) / sigma2) / (math.sqrt(2 * math.pi) * sigma)
norm = Dd * sum(p)
p = p / norm

# randomly sample distribution
M = 200
r = np.random.normal(dbar, sigma, M)
Nb = 26
Db = (dmax - dmin) / (Nb - 1)
bins = dmin + Db * np.arange(Nb)

# cumulative probability
P = Dd * np.cumsum(p)

# build figure
plt.subplots(2,2, figsize=(15,10))

# plot histogram of randomly sampled data
plt.subplot(221)
plt.hist(r, bins)
plt.ylim(0, 40)
plt.xlim(0, 10)
plt.title('Histogram')
plt.xlabel('d')
plt.ylabel('Counts')

# plot probability density function
plt.subplot(222)
plt.plot(d, p)
plt.ylim(0, 0.5)
plt.xlim(0, 10)
plt.title('Probability Density Function')
plt.xlabel('d')
plt.ylabel('p(d)')

# plot probability density function overlain on histogram
plt.subplot(223)
plt.hist(r, bins, normed=True)
plt.plot(d, p)
plt.ylim(0, 0.5)
plt.xlim(0, 10)
plt.title('Histogram and PDF')
plt.xlabel('d')
plt.ylabel('p(d)')

# plot cumulative probability density function
plt.subplot(224)
plt.plot(d, P)
plt.ylim(-0.01, 1.01)
plt.xlim(0, 10)
plt.title('Cumulative PDF')
plt.xlabel('d')
plt.ylabel('P(d)')

# d-axis
Dd = 0.1
N = 101
d = Dd * np.arange(N)

# normal PDF
dbar = 0
sd = 3
p = np.multiply(d, np.exp(-0.5 * np.multiply((d-dbar), (d-dbar)) / sd ** 2))
dbar = 1
sd = 0.3
p = p + 4 * np.exp(-0.5 * np.multiply((d-dbar), (d-dbar)) / sd ** 2)
norm = Dd * sum(p)
p = p / norm

# maximum likelihood point
imax = np.argmax(p)
dmax = d[imax]
dbar = Dd * sum(np.multiply(d, p))

# build plot
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)

ax.plot(d, p)
plt.xlabel('d')
plt.ylabel('p(d)')
plt.title('Probability Density Function')
ax.plot((dmax, dmax), (0, 0.1), 'k-')
ax.annotate('$d_{ML}$', xy=(dmax-0.15, 0.12), size=20)
ax.plot((dbar, dbar), (0, 0.1), 'k-')
ax.annotate(r'$\langle d \rangle$', xy=(dbar-0.2, 0.12), size=20)

# d axis
Dd = 0.1
N = 101
d = Dd * np.arange(N)

# build two normal PDF's
dbar = 5
sd1 = 0.5
p1 = np.exp(-0.5 * np.multiply((d-dbar), (d-dbar)) / sd1 ** 2) / (math.sqrt(2 * math.pi) * sd1)
norm1 = Dd * sum(p1)
sd2 = 1.5
p2 = np.exp(-0.5 * np.multiply((d-dbar), (d-dbar)) / sd2 ** 2) / (math.sqrt(2 * math.pi) * sd2)
norm2 = Dd * sum(p2)

# quadratic
q = np.multiply((d-dbar), (d-dbar))

# products
qp1 = np.multiply(q, p1)
qp2 = np.multiply(q, p2)

# estimate variances
sd21 = Dd * sum(qp1)
sd1est = math.sqrt(sd21)
sd22 = Dd * sum(qp2)
sd2est = math.sqrt(sd22)
print('std dev 1:     true: %s     estimated: %s' % (sd1, sd1est))
print('std dev 2:     true: %s     estimated: %s' % (sd2, sd2est))

# build figure
plt.subplots(2,3, figsize=(15,10))

# plot parabola
plt.subplot(231)
plt.plot(d, q)
plt.title('Parabola')
plt.xlabel('d')
plt.ylabel('q(d)')

# plot PDF
plt.subplot(232)
plt.plot(d, p1)
plt.title('First PDF')
plt.xlabel('d')
plt.ylabel('p(d)')

# plot product (variance)
plt.subplot(233)
plt.plot(d, qp1)
plt.title('Product')
plt.xlabel('d')
plt.ylabel('q(d) p(d)')
plt.fill_between(x=d, y1=np.zeros(qp1.shape), y2=qp1, facecolor='red', alpha=0.1)

# plot parabola
plt.subplot(234)
plt.plot(d, q)
plt.title('Parabola')
plt.xlabel('d')
plt.ylabel('q(d)')

# plot PDF
plt.subplot(235)
plt.plot(d, p2)
plt.title('First PDF')
plt.xlabel('d')
plt.ylabel('p(d)')

# plot product (variance)
plt.subplot(236)
plt.plot(d, qp2)
plt.title('Product')
plt.xlabel('d')
plt.ylabel('q(d) p(d)')
plt.fill_between(x=d, y1=np.zeros(qp2.shape), y2=qp2, facecolor='red', alpha=0.1)

# d axis
Dd = 0.1
N = 101
d1 = Dd * np.arange(N)
d2 = Dd * np.arange(N)
dmin = 0
dmax = 10

# simulate uncorrelated data
d1bar = 5
d2bar = 5
sd1 = 1.25
sd2 = 0.75
cov = 0
C = np.zeros((2,2))
C[0,0] = sd1 ** 2
C[1,1] = sd2 ** 2
C[0,1] = cov
C[1,0] = cov
norm = 2 * math.pi * math.sqrt(np.linalg.det(C))
CI = np.linalg.inv(C)
P1 = np.zeros((N,N))
for i in range(0, N):
    for j in range(0, N):
        dd = np.array([dmin + Dd * i - d1bar, dmin + Dd * j - d2bar])
        P1[i, j] = np.exp(-0.5 * dd.dot(CI).dot(dd)) / norm
        
mycov = 0.5
        
# simulate positively correlated data
C[0,1] = mycov
C[1,0] = mycov
norm = 2 * math.pi * math.sqrt(np.linalg.det(C))
CI = np.linalg.inv(C)
P2 = np.zeros((N,N))
for i in range(0, N):
    for j in range(0, N):
        dd = np.array([dmin + Dd * i - d1bar, dmin + Dd * j - d2bar])
        P2[i, j] = np.exp(-0.5 * dd.dot(CI).dot(dd)) / norm
        
# simulate negatively correlated data
C[0,1] = - mycov
C[1,0] = - mycov
norm = 2 * math.pi * math.sqrt(np.linalg.det(C))
CI = np.linalg.inv(C)
P3 = np.zeros((N,N))
for i in range(0, N):
    for j in range(0, N):
        dd = np.array([dmin + Dd * i - d1bar, dmin + Dd * j - d2bar])
        P3[i, j] = np.exp(-0.5 * dd.dot(CI).dot(dd)) / norm

# build plot
plt.subplots(1,3, figsize=(17,4))

# subplot of PDF of uncorrelated data
plt.subplot(131)
plt.imshow(P1, interpolation='nearest')
plt.grid(True)
plt.colorbar()
plt.xlabel('$d_2$')
plt.ylabel('$d_1$')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.title('PDF of uncorrelated data')
plt.plot((50, 50), (0, 100), 'w--', linewidth=2)
plt.plot((0, 100), (50, 50), 'w--', linewidth=2)
plt.annotate(r'$\langle d_2 \rangle$', xy=(52, 90), xycoords='data', color='white', size=15)
plt.annotate(r'$\langle d_1 \rangle$', xy=(85, 55), xycoords='data', color='white', size=15)

# subplot of PDF of positively correlated data
plt.subplot(132)
plt.imshow(P2, interpolation='nearest')
plt.grid(True)
plt.colorbar()
plt.xlabel('$d_2$')
plt.ylabel('$d_1$')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.title('PDF of positively correlated data')
plt.plot((50, 50), (0, 100), 'w--', linewidth=2)
plt.plot((0, 100), (50, 50), 'w--', linewidth=2)
plt.annotate(r'$\langle d_2 \rangle$', xy=(52, 90), xycoords='data', color='white', size=15)
plt.annotate(r'$\langle d_1 \rangle$', xy=(85, 55), xycoords='data', color='white', size=15)

# subplot of PDF of negatively correlated data
plt.subplot(133)
plt.imshow(P3, interpolation='nearest')
plt.grid(True)
plt.colorbar()
plt.xlabel('$d_2$')
plt.ylabel('$d_1$')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.title('PDF of negatively correlated data')
plt.plot((50, 50), (0, 100), 'w--', linewidth=2)
plt.plot((0, 100), (50, 50), 'w--', linewidth=2)
plt.annotate(r'$\langle d_2 \rangle$', xy=(52, 90), xycoords='data', color='white', size=15)
plt.annotate(r'$\langle d_1 \rangle$', xy=(85, 55), xycoords='data', color='white', size=15)

# d-axis
dd = 0.01
N = 101
d = dd * np.arange(1,N)

# m-axis
dm = 0.01
M = 101
m = dm * np.arange(1,M)

# build uniform distribution for p(d)
pd = np.ones(N-1)

# transform probability density function p(d) to p(m)
J = abs(0.5 / np.sqrt(d))
pm = np.multiply(pd, J)

# build plot
plt.subplots(2,1, figsize=(12,6))
plt.subplots_adjust(hspace=0.5)

plt.subplot(211)
plt.plot(d, pd)
plt.xlabel('d')
plt.ylabel('p(d)')
plt.xlim(0, 1)
plt.ylim(0, 2)
plt.title('PDF of Data')

plt.subplot(212)
plt.plot(m, pm)
plt.xlabel('m')
plt.ylabel('p(m)')
plt.xlim(0, 1)
plt.ylim(0, 2)
plt.title('PDF of Model Transformed from PDF of Data')

# data-axis
dd = 0.1
N = 101
dmin = -5
dmax = 5
d = dmin + dd * np.arange(N)

# first Gaussian distribution
sd_a = 1.0
dbar_a = 0.0
p_a = np.exp(-0.5 * np.multiply(d-dbar_a, d-dbar_a) / sd_a**2) / (math.sqrt(2 * math.pi) * sd_a)

# second Gaussian distribution
sd_b = 2.0
dbar_b = 0.0
p_b = np.exp(-0.5 * np.multiply(d-dbar_b, d-dbar_b) / sd_b**2) / (math.sqrt(2 * math.pi) * sd_b)

# build plot
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)

ax.plot(d, p_a, 'r-', d, p_b, 'b-')
plt.xlabel('d')
plt.ylabel('p(d)')
plt.xlim(dmin, dmax)
plt.ylim(0, 0.5)
plt.title('Gaussian Probability Density Functions')
ax.annotate('(A)', xy=(-0.3, 0.35), size=20)
ax.annotate('(B)', xy=(-0.3, 0.15), size=20)

# data-axis
Dd = 0.1
N = 101
d1 = Dd * np.arange(N)
d2 = Dd * np.arange(N)
dmin = 0
dmax = 10

# first probability density function
d1bar = 4
d2bar = 6
DBAR1 = np.array([d1bar, d2bar])
sd1 = 1.25
sd2 = 0.75
cov = 0.5
C1 = np.zeros((2,2))
C1[0,0] = sd1 ** 2
C1[1,1] = sd2 ** 2
C1[0,1] = cov
C1[1,0] = cov
norm = 2 * math.pi * math.sqrt(np.linalg.det(C1))
C1I = np.linalg.inv(C1)
P1 = np.zeros((N,N))
for i in range(0, N):
    for j in range(0, N):
        dd = np.array([dmin + Dd * i - d1bar, dmin + Dd * j - d2bar])
        P1[i, j] = np.exp(-0.5 * dd.dot(C1I).dot(dd)) / norm

# second probability density function
d1bar = 6
d2bar = 4
DBAR2 = np.array([d1bar, d2bar])
sd1 = 0.75
sd2 = 1.25
cov = -0.5
C2 = np.zeros((2,2))
C2[0,0] = sd1 ** 2
C2[1,1] = sd2 ** 2
C2[0,1] = cov
C2[1,0] = cov
norm = 2 * math.pi * math.sqrt(np.linalg.det(C2))
C2I = np.linalg.inv(C2)
P2 = np.zeros((N,N))
for i in range(0, N):
    for j in range(0, N):
        dd = np.array([dmin + Dd * i - d1bar, dmin + Dd * j - d2bar])
        P2[i, j] = np.exp(-0.5 * dd.dot(C2I).dot(dd)) / norm

# product of the two probability density functions
P3 = np.multiply(P1, P2)

# build plot
plt.subplots(1,3, figsize=(17,4))

# first probability density function
plt.subplot(131)
plt.imshow(P1, interpolation='nearest')
plt.grid(True)
plt.colorbar()
plt.xlabel('$d_2$')
plt.ylabel('$d_1$')
plt.title('PDF1')

# second probability density function
plt.subplot(132)
plt.imshow(P2, interpolation='nearest')
plt.grid(True)
plt.colorbar()
plt.xlabel('$d_2$')
plt.ylabel('$d_1$')
plt.title('PDF2')

# pdf of multiplied probability density functions
plt.subplot(133)
plt.imshow(P3, interpolation='nearest')
plt.grid(True)
plt.colorbar()
plt.xlabel('$d_2$')
plt.ylabel('$d_1$')
plt.title('PDF3 using Product Method')

# CORRECT THEORETICAL DISTRIBUTION

import scipy.stats

# make some Gaussian random data
N = 200
dbar = 5
sd = 1
d = np.random.normal(dbar, sd, N)

# estimate mean and standard deviation
dbar_est = np.mean(d)
sd_est = np.std(d)
print('Estimated mean = ', dbar_est, ', and Estimated standard deviation = ', sd_est)

# calculate and normalize histogram
h = np.histogram(d, 40)
pdf_est = h[0] / sum(h[0])

# get the theoretical distribution
pdfc_tru = norm.cdf(h[1], dbar_est, sd_est)[1:41] - norm.cdf(h[1], dbar_est, sd_est)[0:40]

# compute Chi-squared statistic
chi2_est = N * sum(np.divide(np.multiply(pdf_est - pdfc_tru, pdf_est - pdfc_tru), pdfc_tru))
p_est = 1 - scipy.stats.chi2.cdf(chi2_est, 40-3)
print(chi2_est)
print(p_est)

# INCORRECT THEORETICAL DISTRIBUTION

# estimate mean and standard deviation of our data incorrectly (mathematically adjust them)
dbar_est = np.mean(d) - 0.5
sd_est = np.std(d) * 1.5

# get the (incorrect) theoretical distribution
pdfi_tru = norm.cdf(h[1], dbar_est, sd_est)[1:41] - norm.cdf(h[1], dbar_est, sd_est)[0:40]

# compute Chi-squared statistic
chi2_est = N * sum(np.divide(np.multiply(pdf_est - pdfi_tru, pdf_est - pdfi_tru), pdfi_tru))
p_est = 1 - scipy.stats.chi2.cdf(chi2_est, 40-3)
print(chi2_est)
print(p_est)

# build figure
plt.subplots(1,2, figsize=(12,4))

# make correct distribution graph
plt.subplot(121)
plt.plot(h[1][0:40], pdf_est)
plt.plot(h[1][0:40], pdfc_tru)
plt.title('Correct Theoretical Distribution')

# make incorrect distribution graph
plt.subplot(122)
plt.plot(h[1][0:40], pdf_est)
plt.plot(h[1][0:40], pdfi_tru)
plt.title('Incorrect Theoretical Distribution')

# setup vectors d1 and d2
L = 40
Dd = 1.0
d1 = Dd * np.arange(L)
d2 = Dd * np.arange(L)

# make a Gaussian distribution
d1bar = 15
d2bar = 25
s1 = 7
s2 = 8
norm = 1 / (2 * math.pi * s1 * s2)
p1 = np.exp(-(np.multiply(d1-d1bar, d1-d1bar) / (2 * s1**2)))
p2 = np.exp(-(np.multiply(d2-d2bar, d2-d2bar) / (2 * s2**2)))
P = norm * p1.reshape(L,1).dot(p2.reshape(1,L))

# sum along cols, integrating P along d2 to get p1=p(d1)
p1 = Dd * np.sum(P, axis=1)

# sum along rows, integrating P along d1 to get p2=p(d2)
p2 = Dd * np.sum(P, axis=0)

# conditional distribution P1g2 = P(d1|d2) = P(d1,d2) / p2
P1g2 = np.divide(P, np.ones(L).reshape(L,1) * p2)

# conditional distribution P2g1 = P(d2|d1) = P(d1,d2) / p1
P2g1 = np.divide(P, p1.reshape(L,1) * np.ones(L))

# build plot
plt.subplots(1,3, figsize=(17,4))

# Gaussian Joint PDF p(d1,d2)
plt.subplot(131)
plt.imshow(P, interpolation='nearest')
plt.grid(True)
plt.colorbar()
plt.xlabel('$d_2$')
plt.ylabel('$d_1$')
plt.title('Gaussian Joint PDF $p(d_1,d_2)$')

# Conditional PDF p(d2|d1)
plt.subplot(132)
plt.imshow(P1g2, interpolation='nearest')
plt.grid(True)
plt.colorbar()
plt.xlabel('$d_2$')
plt.ylabel('$d_1$')
plt.title('Conditional PDF $p(d_2|d_1)$')

# Conditional PDF p(d1|d2)
plt.subplot(133)
plt.imshow(P2g1, interpolation='nearest')
plt.grid(True)
plt.colorbar()
plt.xlabel('$d_2$')
plt.ylabel('$d_1$')
plt.title('Conditional PDF $p(d_1|d_2)$')

P = np.array([(0.8, 0.001), (0.1, 0.099)])
print(P)

# set mean and variance for simulated data
N = 1000
mbar = 5.0
sigma = 1.0

# randomly sample using Gaussian distribution
mnormal1 = np.random.normal(mbar, sigma, N)

# randomly sample using uniform distribution and transform
muniform = np.random.uniform(0, 1, N)
mnormal2 = scipy.stats.norm.ppf(muniform, loc=mbar, scale=sigma)

# build figure
plt.subplots(1,2, figsize=(15,6))

# plot histogram of randomly sampled data
plt.subplot(121)
plt.hist(mnormal1, normed=True)
plt.title('Histogram of Random Gaussian')
plt.xlim(0,10)
plt.ylim(0,0.5)
plt.xlabel('d')
plt.ylabel('p(d)')

# plot probability density function
plt.subplot(122)
plt.hist(mnormal2, normed=True)
plt.title('Histogram of Transformed Gaussian')
plt.xlim(0,10)
plt.ylim(0,0.5)
plt.xlabel('d')
plt.ylabel('p(d)')

# Create realizations of an exponential probability density function in two ways:
#      A) Transformation of a Uniform distribution
#      B) Metropolis-Hastings algorithm
# In this example, pdf is p(d) = c*exp(-d)/c; for d > 0

# setup data vector
dmin = -10
dmax = 10
N = 201
Dd = (dmax - dmin) / (N-1)
d = dmin + Dd * np.arange(N)

# evaluate exponential distribution
c = 2.0
pexp = (0.5 / c) * np.exp(-abs(d) / c)

# A) Transformation of a Uniform distribution
M = 5000
rm = np.random.uniform(-1, 1, M)
rd1 = np.multiply(- np.sign(rm) , c * np.log((1-np.absolute(rm))))

# B) Metropolis-Hastings algorithm
Niter = 5000
rd2 = np.zeros(Niter)
prd = np.zeros(Niter)
rd2[0] = 0.0
prd[0] = (1/c) * np.exp(- np.absolute(rd2[1]) / c)
s = 1

for k in range(1, Niter):
    # old realization
    rdo = rd2[k-1]
    prdo = prd[k-1]
    rdn = np.random.normal(rdo, s, 1)
    prdn = (0.5 / c) * math.exp(-abs(rdn)/c)
    
    alpha = prdn / prdo
    
    if (alpha > 1):
        rd2[k] = rdn
        prd[k] = prdn
    else:
        r = np.random.uniform(0, 1, 1)[0]
        if (alpha > r):
            rd2[k] = rdn
            prd[k] = prdn
        else:
            rd2[k] = rdo
            prd[k] = prdo

# build figure
plt.subplots(1,2, figsize=(15,6))

# plot histogram of randomly sampled data
plt.subplot(121)
plt.hist(rd1, bins=30, normed=True)
plt.plot(d, pexp)
plt.title('Histogram of Transformed Exponential')
plt.xlim(-10,10)
plt.ylim(0,0.3)
plt.xlabel('d')
plt.ylabel('p(d)')

# plot histogram of randomly sampled data
plt.subplot(122)
plt.hist(rd2, bins=30, normed=True)
plt.plot(d, pexp)
plt.title('Histogram of Metropolis-Hastings Exponential')
plt.xlim(-10,10)
plt.ylim(0,0.3)
plt.xlabel('d')
plt.ylabel('p(d)')

