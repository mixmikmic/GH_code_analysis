import matplotlib.pyplot as plt
import numpy as np
import nibabel
import h5py
import os

# For to fancy
import fakedata as fd
import cortex as cx

# For to machine learn
from sklearn import svm
from sklearn import discriminant_analysis as da

get_ipython().run_line_magic('matplotlib', 'inline')

# Answer


# Define 2D values for class 1
n = 500 # number of exemplars per class
#r = 0.6
mu_c1 = np.array([-0.5, 0.3])
mu_c2 = np.array([0.5, -0.3])
Xt, Xv, Yt, Yv = generate_data(n, mu_c1, mu_c2) #, cov_c1=r, cov_c2=r)

plot_classes(Xt[Yt==1], Xt[Yt==2])

# Get SVM classifier base object
svmclf = svm.LinearSVC()
# Fit SVM classifier to data
_ = svmclf.fit(Xt, Yt)
# Predict new values
Ypred_svm = svmclf.predict(Xv)

ldaclf = da.LinearDiscriminantAnalysis()
# Fit LDA classifier to data
_ = ldaclf.fit(Xt, Yt)
# Predict new responses w/ LDA classifier
Ypred_lda = ldaclf.predict(Xv)

# Answer

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


subject = 's03'
transform = 'category_localizer'
roi_masks = cx.get_roi_masks('s03', 'category_localizer', roi_list=['V1','V2','V3','V4','LO','OFA','FFA','EBA',
                                                                   'PPA','RSC','OPA'])
all_masks = np.array(list(roi_masks.values()))
print(all_masks.shape)
mask = np.any(all_masks, axis=0)
print(mask.shape)
#cx.webgl.show(cx.Volume(mask, subject, transform))

# Answer


fdir = '/unrshare/LESCROARTSHARE/IntroToEncodingModels/'
fbase = os.path.join(fdir, 's03_catloc_run%02d.nii.gz')
data = []
for run in range(1, 7):
    nii = nibabel.load(fbase%run)
    # Transpose at load time to make the data [t, z, y, x]
    tmp = nii.get_data().T
    # Mask the data to select only the voxels we care about
    data.append(tmp[:, mask])

# Answer

with h5py.File(os.path.join(fdir, 'catloc_design.hdf')) as hf:
    print(list(hf.keys()))
    X = hf['X'].value
    xnames = hf['xnames'].value.tolist()
    # Ignore the 'decode' for now, it has to do with the format in which the strings were stored in this 
    # file, and it's just confusing...
    class_names = ['null'] + [x.decode() for x in xnames]
    events = hf['events'].value

# Answer
# (Display each variable, figure out what each is!)

# Answer 

# Answer

from sklearn import metrics

cmatrix = metrics.confusion_matrix(Yv, Yv_pred)

# Show matrix
fig, ax = plt.subplots()
im = ax.imshow(cmatrix)
ax.set_xticks(np.arange(6))
ax.set_yticks(np.arange(6))
ax.set_xticklabels(class_names, rotation=90)
ax.set_yticklabels(class_names)
plt.colorbar(im)

# Answer

# Answer



