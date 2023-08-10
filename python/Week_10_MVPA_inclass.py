import matplotlib.pyplot as plt
import numpy as np
import nibabel
import h5py
import os

# For to fancy
#import fakedata as fd
import cortex as cx

# For to machine learn
from sklearn import svm
from sklearn import discriminant_analysis as da

get_ipython().run_line_magic('matplotlib', 'inline')

# Answer
n = 300
mu_c1 = np.array([3, 1])
mu_c2 = np.array([-1, 0])
X1 = np.random.randn(n, 2) + mu_c1
X2 = np.random.randn(n, 2) + mu_c2
plt.scatter(X1[:, 0], X1[:, 1], alpha=0.3)
plt.scatter(X2[:, 0], X2[:, 1], alpha=0.3)

Y1 = np.ones((n,))
Y2 = np.ones((n,)) * 2
X = np.vstack([X1, X2])
Y = np.hstack([Y1, Y2])



# Answer (fancy)
def generate_data(n, mu_c1, mu_c2, cov_c1=0, cov_c2=0, trn_frac=0.8):
    """For 2 classes only, 2D variables
    
    Parameters
    ----------
    n is number per class
    corr_mag is correlation btw 1st and 2nd dim
    slope is magnitude of 1st dim / 2nd dim
    mu_c1 and mu_c2 are means for each dimension of X1 and X2"""
    if cov_c1 is None:
        cov_c1 = np.array([[1, 0], [0, 1]])
    elif not isinstance(cov_c1, np.ndarray):
        cov_c1 = np.array([[1, cov_c1], [cov_c1, 1]])
    if cov_c2 is None:
        cov_c2 = np.array([[1, 0], [0, 1]])
    elif not isinstance(cov_c2, np.ndarray):
        cov_c2 = np.array([[1, cov_c2], [cov_c2, 1]])
    # First variable
    X1 = np.random.multivariate_normal(mu_c1, cov_c1, size=(n,))
    # Second variable
    X2 = np.random.multivariate_normal(mu_c2, cov_c2, size=(n,))    
    nn = np.int(n*trn_frac)
    Xt = np.vstack([X1[:nn], X2[:nn]])
    Xv = np.vstack([X1[nn:], X2[nn:]])
    Yt = np.hstack([np.ones(nn), np.ones(nn)*2])
    Yv = np.hstack([np.ones((n-nn,)), np.ones((n-nn,))*2])
    return Xt, Xv, Yt, Yv

def plot_classes(X1, X2, classifier=None, axis=(-5, 5, -5, 5), clfcolor='k', plot_data=True, ax=None):
    """For now: only linear classifier, only w/ 2D plots"""
    if ax is None:
        fig, ax = plt.subplots()
    if plot_data:
        ax.scatter(X1[:,0], X1[:,1], color='r', alpha=0.3)
        ax.plot(*mu_c1, color='r', marker='o', markeredgecolor='k')
        ax.scatter(X2[:,0], X2[:,1], color='b', alpha=0.3)
        ax.plot(*mu_c2, color='b', marker='o', markeredgecolor='k')
        ax.axis(axis)
    if classifier is not None:
        # useful: https://stackoverflow.com/questions/22294241/plotting-a-decision-boundary-separating-2-classes-using-matplotlibs-pyplot
        w = classifier.coef_[0]
        # Slope (m from y = mx + b)
        m = -w[0] / w[1]
        # Get intercept (b from y = mx + b)
        b = - (classifier.intercept_[0]) / w[1]
        # Sample some X values, and compute the corresponding Ys
        xx = np.linspace(-5, 5)
        yy = m * xx + b
        ax.plot(xx, yy, clfcolor+'-')

Yt.shape

# Define 2D values for class 1
n = 500 # number of exemplars per class
#r = 0.6
mu_c1 = np.array([-1.5, 1.3])
mu_c2 = np.array([0.5, -0.3])
Xt, Xv, Yt, Yv = generate_data(n, mu_c1, mu_c2) #, cov_c1=r, cov_c2=r)

plot_classes(Xt[Yt==1], Xt[Yt==2])

svmclf = svm.LinearSVC()

_ = svmclf.fit(Xt, Yt)

# Get SVM classifier base object
svmclf = svm.LinearSVC()
# Fit SVM classifier to data
_ = svmclf.fit(Xt, Yt)
# Predict new values
Ypred_svm = svmclf.predict(Xv)

Ypred_svm

ldaclf = da.LinearDiscriminantAnalysis()
# Fit LDA classifier to data
_ = ldaclf.fit(Xt, Yt)
# Predict new responses w/ LDA classifier
Ypred_lda = ldaclf.predict(Xv)

# Answer
svm_acc = (Yv==Ypred_svm).mean()
lda_acc = (Yv==Ypred_lda).mean()
print(svm_acc, lda_acc)

svmclf.coef_, svmclf.intercept_

# Ax + By + C form:
print(ldaclf.coef_) # [A, B]
print(ldaclf.intercept_) # [C]

# Show slope & intercept of decision plane line for lda
print('Slope = %0.2f'%(- ldaclf.coef_[0][0]/ldaclf.coef_[0][1]))
print('Intercept = %0.2f'%(- ldaclf.intercept_[0]/ldaclf.coef_[0][1]))

# Show slope & intercept of decision plane line for svm
print('Slope = %0.2f'%(- svmclf.coef_[0][0]/svmclf.coef_[0][1]))
print('Intercept = %0.2f'%(- svmclf.intercept_[0]/svmclf.coef_[0][1]))

fig, ax = plt.subplots()
plot_classes(Xt[Yt==1], Xt[Yt==2], classifier=svmclf, clfcolor='c-', ax=ax)
plot_classes(Xt[Yt==1], Xt[Yt==2], classifier=ldaclf, clfcolor='m-', plot_data=False, ax=ax)

fig, ax = plt.subplots()
plot_classes(Xv[Yv==1], Xv[Yv==2], classifier=svmclf, clfcolor='c-', ax=ax)
plot_classes(Xv[Yv==1], Xv[Yv==2], classifier=ldaclf, clfcolor='m-', plot_data=False, ax=ax)

# Answer


# Implement permutation! 1:200
acc = []
for rpt in range(1000):
    nshuf = len(Yt)
    ridx = np.random.permutation(np.arange(0, nshuf, 1))
    Yt_rand = Yt[ridx]
    #Yt_rand
    # Get SVM classifier base object
    svmclf_rnd = svm.LinearSVC()
    # Fit SVM classifier to data
    _ = svmclf_rnd.fit(Xt, Yt_rand)
    # Predict new values
    Ypred_svm_rand = svmclf_rnd.predict(Xv)
    acc.append(np.mean(Ypred_svm_rand==Yv))

plt.hist(acc)



subject = 's03'
transform = 'category_localizer'
roi_masks = cx.get_roi_masks('s03', 'category_localizer', roi_list=['V1','V2','V3','V4',
                                                                    'LO','OFA','FFA','EBA',
                                                                   'PPA','RSC','OPA'])
all_masks = np.array(list(roi_masks.values()))
print(all_masks.shape)
mask = np.any(all_masks, axis=0)
print(mask.shape)
#cx.webgl.show(cx.Volume(mask, subject, transform))

mask.sum()

# Answer
V = cx.Volume(mask, subject, transform, vmin=0, vmax=1, cmap='gray')
h = cx.webgl.show(V, open_browser=False)

from scipy.stats import zscore

fdir = '/unrshare/LESCROARTSHARE/IntroToEncodingModels/'
fbase = os.path.join(fdir, 's03_catloc_run%02d.nii.gz')
data = []
for run in range(1, 7):
    nii = nibabel.load(fbase%run)
    # Transpose at load time to make the data [t, z, y, x]
    tmp = nii.get_data().T
    # Mask the data to select only the voxels we care about
    data.append(zscore(tmp[:, mask], axis=0))

data[0].shape

# Answer
Xt = np.vstack(data[:4])
Xv = np.vstack(data[4:])

Xt.shape

with h5py.File(os.path.join(fdir, 'catloc_design.hdf')) as hf:
    print(list(hf.keys()))
    X = hf['X'].value
    xnames = hf['xnames'].value.tolist()
    # Ignore the 'decode' for now, it has to do with the format in which the strings were stored in this 
    # file, and it's just confusing...
    class_names = ['null'] + [x.decode() for x in xnames]
    events = hf['events'].value

len(Xt) + len(Xv)

events

720/6

n_tr_per_run = 120
n_runs = 4
Yt = events[:n_tr_per_run*n_runs]
Yv = events[n_tr_per_run*n_runs:]



# Answer
plt.imshow(X, aspect='auto')
# (Display each variable, figure out what each is!)

# Answer 

# Answer
# Get SVM classifier base object
svmclf = svm.LinearSVC()
# Fit SVM classifier to data
_ = svmclf.fit(Xt, Yt)
# Predict new values
Ypred_svm = svmclf.predict(Xv)

np.mean(Ypred_svm==Yv)

from sklearn import metrics

cmatrix = metrics.confusion_matrix(Yv, Ypred_svm)

# Show matrix
fig, ax = plt.subplots()
im = ax.imshow(cmatrix)
ax.set_xticks(np.arange(6))
ax.set_yticks(np.arange(6))
ax.set_xticklabels(class_names, rotation=90)
ax.set_yticklabels(class_names)
plt.colorbar(im)

# Answer
svmclf.coef_.shape

np.unique(Yv)

svmclf.coef_.max()

# Answer
clf_wts = dict()
for i, cname in enumerate(class_names):
    clf_wts[cname] = cx.Volume(svmclf.coef_[i, :], subject, transform, 
                                        mask=mask, vmin=-0.05, vmax=0.05)

cx.webgl.show(clf_wts, open_browser=False)





# Answer



