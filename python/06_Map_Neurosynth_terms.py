get_ipython().magic('matplotlib inline')

from nilearn.datasets import load_mni152_template
from nilearn import plotting, image
import nibabel as nib

import statsmodels.api as sm
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 18

import numpy as np
import glob

tfiles = glob.glob('../data/Neurosynth_topic_maps/*.nii.gz')

topicname = np.array(['_'.join(t.split('/')[-1].split('_')[2:5]) for t in tfiles])
np.savetxt('../data/cifti/topics.txt', topicname, fmt = '%s')

gradients = nib.load('../data/rsFC_eigenvectors.dscalar.nii').get_data().T
gradients = sm.add_constant(gradients[:,0:3])

topics = nib.load('../data/cifti/topics25.dscalar.nii').get_data().T
topicname = np.genfromtxt('../data/cifti/topics.txt', dtype = str)

coefs = np.zeros([topics.shape[1],5])

print "Topic                                         G1         G2          G3        F         p"
print "==============================================================================================="
for i in range(topics.shape[1]): 
    ols = sm.OLS(topics[:,i], gradients)
    ols = ols.fit(cov_type="HAC", cov_kwds={'maxlags':1000})
    #ols_null = sm.OLS(topics[:,i], np.ones(gradients.shape[0]))
    #ols_null = ols_null.fit(cov_type="HAC", cov_kwds={'maxlags':5000})
    coefs[i,0:3] = ols.params[1:4]
    coefs[i,3] = ols.fvalue
    coefs[i,4] = ols.f_pvalue
    print "%-40s      %-6.2f     %-6.2f      %-6.2f    %-6.2f    %-6.4f" % (topicname[i], 
                                                                  ols.params[1], ols.params[2], ols.params[3],
                                                                  ols.fvalue, ols.f_pvalue)

max_coefs = np.max(coefs[:,0:3], axis = 1)
good = np.where(np.logical_or(max_coefs > 0.05, max_coefs < -0.05))[0]

topicname[good]

topicname_clean = ['action','object',
                   'performance','sensory',
                   'face', 'inhibition',
                   'perception','retrieval',
                   'eye','social',
                   'language','working_memory',
                   'reward','auditory',
                   'somatosensory','comprehension', 
                   'emotion', 'attention', 
                   'number','semantic',
                   'response_conflict','pain',
                   'movement']

zip(topicname[good], topicname_clean)

coefs_thr = coefs[good,:]
topicname_thr = topicname[good]

sns.set_style('white')

fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111)
#ax.scatter(coefs_thr[:,0], coefs_thr[:,1], s = 100, c = 'red')

for i in range(len(topicname_clean)):
    ax.annotate(topicname_clean[i], xy=(coefs_thr[i,0], coefs_thr[i,1]), 
                xytext=(coefs_thr[i,0], coefs_thr[i,1]),
                #arrowprops=dict(facecolor='black', shrink=0.05),
                )


ax.set_xlim(-0.6,0.6)
ax.set_ylim(-0.6,0.6)

ax.set_title('Gradients 1 & 2', fontsize = 24)
ax.set_xlabel('Gradient 1', fontsize = 18)
ax.set_ylabel('Gradient 2', fontsize = 18)

plt.tight_layout()

plt.savefig('../figures/terms_G12.pdf')

sns.set_style('white')

fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111)

for i in range(len(topicname_clean)):
    ax.annotate(topicname_clean[i], xy=(coefs_thr[i,0], coefs_thr[i,2]), 
                xytext=(coefs_thr[i,0], coefs_thr[i,2]),
                #arrowprops=dict(facecolor='black', shrink=0.05),
                )

ax.set_ylim(-0.5,0.5)
ax.set_xlim(-0.5,0.5)

ax.set_title('Gradients 1 & 3', fontsize = 24)
ax.set_xlabel('Gradient 1', fontsize = 18)
ax.set_ylabel('Gradient 3', fontsize = 18)

plt.tight_layout()

plt.savefig('../figures/terms_G13.pdf')

sns.set_style('white')

fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111)
#ax.scatter(coefs_thr[:,0], coefs_thr[:,1], s = 100, c = 'red')

for i in range(len(topicname_clean)):
    ax.annotate(topicname_clean[i], xy=(coefs_thr[i,1], coefs_thr[i,2]), 
                xytext=(coefs_thr[i,1], coefs_thr[i,2]),
                #arrowprops=dict(facecolor='black', shrink=0.05),
                )


ax.set_xlim(-0.6,0.6)
ax.set_ylim(-0.6,0.6)

ax.set_title('Gradients 2 & 3', fontsize = 24)
ax.set_xlabel('Gradient 2', fontsize = 18)
ax.set_ylabel('Gradient 3', fontsize = 18)

plt.tight_layout()

plt.savefig('../figures/terms_G23.pdf')

