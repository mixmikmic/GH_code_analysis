import cv2

get_ipython().magic('matplotlib notebook')
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("dark")

training_bgr = cv2.imread('data/skin-training.jpg')
training_rgb = cv2.cvtColor(training_bgr, cv2.COLOR_BGR2RGB)
training = cv2.cvtColor(training_bgr, cv2.COLOR_BGR2LAB)
M, N, _ = training.shape

mask = np.zeros((M,N))
mask[training[:,:,0] > 160] = 1

plt.subplot(1,2,1)
plt.imshow(training_rgb)
plt.subplot(1,2,2)
plt.imshow(mask, cmap=plt.cm.binary_r)

data = training.reshape(M*N, -1)[:,1:]
data

target = mask.reshape(M*N)
target

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(data, target)

test_bgr = cv2.imread('data/thiago.jpg')
test_rgb = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2RGB)
test = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2LAB)
M_tst, N_tst, _ = test.shape

data = test.reshape(M_tst * N_tst, -1)[:,1:]
skin_pred = gnb.predict(data)
S = skin_pred.reshape(M_tst, N_tst)

plt.subplot(1,3,1)
plt.imshow(test_rgb)
plt.subplot(1,3,2)
plt.imshow(S, cmap=plt.cm.binary_r)
plt.subplot(1,3,3)
plt.imshow(test_rgb, alpha=0.6)
plt.imshow(S, cmap=plt.cm.binary_r, alpha=0.4)

I = cv2.imread('data/BSD-118035.jpg')
I_Lab = cv2.cvtColor(I, cv2.COLOR_BGR2LAB)
h, w, _ = I_Lab.shape
from sklearn.cluster import MeanShift, estimate_bandwidth
X = I_Lab.reshape(h*w, -1)
X

b = estimate_bandwidth(X, quantile=0.1, n_samples=2500)
ms = MeanShift(bandwidth=b, bin_seeding=True)
ms.fit(X)

S = np.zeros_like(I)
L = ms.labels_.reshape(h, w)
num_clusters = ms.cluster_centers_.shape[0]
print num_clusters

for c in range(num_clusters):
    S[L == c] = ms.cluster_centers_[c]

plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(S, cv2.COLOR_LAB2RGB))

from skimage.color import label2rgb 
segments = label2rgb(L)
plt.subplot(1,3,3)
plt.imshow(segments)

frames = get_ipython().getoutput('ls data/CAVIAR_LeftBag/*.jpg')
# Let's find the frame dimensions, M x N
F = cv2.imread(frames[0], cv2.IMREAD_GRAYSCALE)
M, N = F.shape
T = len(frames)
M, N, T

num_pixels = M * N
V = np.zeros((num_pixels, T), dtype=np.float)

for t, fname in enumerate(frames):
    F = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    V[:,t] = np.array(F, dtype=float).reshape(-1)/255
    
V.shape

print V[num_pixels/2]

hist = plt.hist(V[num_pixels/2], bins=20)
plt.title('Central pixel values')

plt.subplot(1,2,1)
plt.imshow(V[:,0].reshape(M, N), cmap=plt.cm.gray)

plt.subplot(1,2,2)
plt.imshow(V[:,100].reshape(M, N), cmap=plt.cm.gray)

from sklearn.mixture import GaussianMixture
K = 3

pixels = range(num_pixels)
gmm = [GaussianMixture(n_components=K) for p in pixels]

for p in pixels:
    gmm[p].fit(V[p,600:750].reshape(-1,1))

bg_mean = np.array([model.means_ for model in gmm])
bg_weight = np.array([model.weights_ for model in gmm])

for k in range(K):
    plt.subplot(1, K, k+1)   
    mu = bg_mean[:,k].reshape(M, N)
    plt.imshow(mu, cmap=plt.cm.gray)
    plt.title('Gaussian model %d' % k)

t = 100
# Find the selected Gaussian model for each pixel p
c = np.array([gmm[p].predict(V[p,t]) for p in pixels])

model_mean = np.array([gmm[p].means_[c[p]] for p in pixels]).reshape(num_pixels, 1)
model_var = np.array([gmm[p].covariances_[c[p]] for p in pixels]).reshape(num_pixels, 1)
model_stddev = np.sqrt(model_var)

is_weight_enough = np.array([bg_weight[p, c[p]] for p in pixels]) > 0.01
is_inlier = np.abs(V[:,t].reshape(-1,1) - model_mean) < 12. * model_stddev

background = is_weight_enough & is_inlier

plt.subplot(1,2,1)
plt.imshow(V[:,t].reshape(M, N), cmap=plt.cm.gray)

plt.subplot(1,2,2)
plt.imshow(background.reshape(M, N), cmap=plt.cm.binary)

