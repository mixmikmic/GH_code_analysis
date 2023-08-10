# Imports
import os
import glob
import numpy as np
import pandas as pd
import nilearn as nil
import nibabel as nib
import brainbox as bb

import multiprocessing as mp
import statsmodels.api as sm
from scipy import stats as st
from matplotlib import gridspec
from scipy import cluster as scl
from nilearn import plotting as nlp

from matplotlib import pyplot as plt
from sklearn import linear_model as slin
import matplotlib.animation as animation
from statsmodels.sandbox import stats as sts
from matplotlib.colors import LinearSegmentedColormap

get_ipython().magic('pylab inline')

def remap(vec, mask):
    # Remap the map into volume space
    vol = np.zeros_like(mask, dtype=np.float64)
    vol[mask] = vec
    
    return vol

def make_image(vec, mask_i):
    mask = mask_i.get_data() != 0
    vol = remap(vec, mask)
    img = nib.Nifti1Image(vol, affine=mask_i.get_affine(), header=mask_i.get_header())
    return img, vol

# Define a new colormap
cdict = {'red':   ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 1.0, 1.0),
                   (0.25, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (0.75, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),

         'blue':  ((0.0, 1.0, 1.0),
                   (0.25, 1.0, 1.0),
                   (0.5, 0.0, 0.0),
                   (1.0, 0.0, 0.0))
        }
hotcold = LinearSegmentedColormap('hotcold', cdict)

# Paths
mask_path = '/data1/abide/Mask/mask_data_specific.nii.gz'
prior_path = '/data1/cambridge/template/template_cambridge_basc_multiscale_sym_scale012.nii.gz'
pheno_path = '/data1/abide/Pheno/merged_pheno.csv'
sub_path = '/data1/abide/Full/abide_release_sym_gsc0_lp01/Stanford/fmri_0051198_session_1_run1.nii.gz'

# Get the mask
m_img = nib.load(mask_path)
mask = m_img.get_data()!=0

# Get the subject
s_img = nib.load(sub_path)
data = s_img.get_data()[mask]

# Get the prior
p_img = nib.load(prior_path)
prior = p_img.get_data()
part = prior[mask]

# Make an image of just network 8
net_8 = np.zeros_like(prior)
net_8[prior==8] = 1
net_8_img = nib.Nifti1Image(net_8, affine=m_img.get_affine(), header=m_img.get_header())

# Make a mean time series for network 8
net_data = data[part==8]
net_mean = np.mean(net_data,0)

# Make some sliding windows
n_img = data.shape[1]
sld_width = 80
window = np.arange(0,n_img-sld_width)

def get_net(w_id, windows, data, sld_width, part):
    # Get the correlation of the network average with the rest of the brain in the sliding window
    w_start = window[w_id]
    w_stop = w_start+sld_width
    sld_data = data[:,w_start:w_stop]

    n_img = sld_data.shape[1]
    n_vox = sld_data.shape[0]
    # Get the network average for all networks
    net_avg = np.zeros((12, n_img))
    net_corr = np.zeros((12, n_vox))
    for n_id, n_val in enumerate(np.arange(1,13)):
        net_avg[n_id] = np.mean(sld_data[part==n_val,...],0)
        net_corr[n_id, :] = np.array([np.corrcoef(sld_data[i,:],net_avg[n_id,:])[0,1] for i in np.arange(n_vox)])
    
    bin_part = np.argmax(net_corr,0)
    
    return net_avg, net_corr, bin_part

s1 = 10
s2 = 70
s3 = 130

def run_par(args):
    w, window, data, sld_width, part = args
    a1, c1, p1 = get_net(w, window, data, sld_width, part)
    np1 = np.zeros(part.shape)
    np1[p1==7] = 1
    out = make_image(np1, m_img)[1]
    
    return out

job_list = list()
for w_id, w in enumerate(window):
    job_list.append((w, window, data, sld_width, part))

p = mp.Pool(processes=7)
results = p.map(run_par, job_list)

net1 = np.zeros_like(mask, dtype=float)
for res in results:
    net1 += res
net1 = net1/len(res)

ni_avg1 = nib.Nifti1Image(net1, affine=m_img.get_affine(), header=m_img.get_header())

nlp.plot_glass_brain(ni1)

a1, c1, p1 = get_net(s1, window, data, sld_width, part)
np1 = np.zeros_like(part)
np1[p1==7] = 1
ni1 = make_image(np1, m_img)
ci1 = make_image(c1[7,:], m_img)

a2, c2, p2 = get_net(s2, window, data, sld_width, part)
np2 = np.zeros_like(part)
np2[p2==7] = 1
ni2 = make_image(np2, m_img)
ci2 = make_image(c2[7,:], m_img)

a3, c3, p3 = get_net(s3, window, data, sld_width, part)
np3 = np.zeros_like(part)
np3[p3==7] = 1
ni3 = make_image(np3, m_img)
ci3 = make_image(c3[7,:], m_img)

# Make a static figure
fig = plt.figure(figsize=(14,8))
gs = gridspec.GridSpec(2, 3, hspace=0.6)

ax_prior = fig.add_subplot(gs[0,0])
ax_prior.set_title('Prior')

ax_time = fig.add_subplot(gs[0,1])
ax_time.plot(net_mean)
ax_time.set_xlim([0,220])
ax_time.set_xlabel('time')
ax_time.set_ylabel('network signal')
ax_time.set_title('Average signal in network 8')
ax_time.set_yticklabels([])

ax_stab = fig.add_subplot(gs[0,2])
ax_stab.set_title('Stability map')

ax_corr = fig.add_subplot(gs[1,:])
ax_corr1.set_title('seed window 1')

def fig_iter(i, window, ax):
    if i%2 == 0:
        ax.axvspan(window[i], window[i]+80, color='red', alpha=0.5)
    else:
        ax.axvspan(window[i], window[i]+80, color='blue', alpha=0.5)
    ax.set_title('{}'.format(i))
    return ax

from tempfile import NamedTemporaryFile

VIDEO_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""

def anim_to_html(anim):
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=20, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")
    
    return VIDEO_TAG.format(anim._encoded_video)

from IPython.display import HTML

def display_animation(anim):
    plt.close(anim._fig)
    return HTML(anim_to_html(anim))

n_img = data.shape[1]
sld_width = 80
window = np.arange(0,n_img-sld_width)
n_iter = len(window)

window.shape

# Make some sliding windows
n_img = data.shape[1]
sld_width = 80
window = np.arange(0,n_img-sld_width)
n_iter = len(window)

fig = plt.figure(figsize=(14,8))
gs = gridspec.GridSpec(1,1, hspace=0.6)

ax = fig.add_subplot(gs[0,0])
ax.set_xlim([0,220])
ax.set_xlabel('time')
ax.set_ylabel('network signal')

ims = []
for i in np.arange(40):
    a = ax.axvspan(window[i], window[i]+80, color='red', alpha=0.5)
    ims.append(ax)

im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000,
    blit=True)

#line_ani = animation.FuncAnimation(fig, fig_iter, 40, fargs=(window, ax),
#    interval=1, blit=True)
display_animation(im_ani)

a.draw()

a

# Make a static figure
fig = plt.figure(figsize=(14,8))
gs = gridspec.GridSpec(3, 3, hspace=0.6)
ax_prior = fig.add_subplot(gs[0,0])
nlp.plot_stat_map(net_8_img, axes=ax_prior, colorbar=False, display_mode='x', cut_coords=(0,), cmap=cm.RdBu_r)
ax_prior.set_title('Prior')

ax_time = fig.add_subplot(gs[0,1])
ax_time.plot(net_mean)
ax_time.axvspan(window[s1], window[s1]+sld_width, color='red', alpha=0.5)
ax_time.axvspan(window[s2], window[s2]+sld_width, color='yellow', alpha=0.5)
ax_time.axvspan(window[s3], window[s3]+sld_width, color='blue', alpha=0.5)
ax_time.set_xlim([0,220])
ax_time.set_xlabel('time')
ax_time.set_ylabel('network signal')
ax_time.set_title('Average signal in network 8')
ax_time.set_yticklabels([])

ax_stab = fig.add_subplot(gs[0,2])
nlp.plot_stat_map(ni_avg1, axes=ax_stab, colorbar=False, display_mode='x', cut_coords=(0,), cmap=hotcold)
ax_stab.set_title('Stability map')

ax_corr1 = fig.add_subplot(gs[1,0])
nlp.plot_glass_brain(ci1[0], cmap=hotcold, axes=ax_corr1, vmin=-1, vmax=1)
ax_corr1.set_title('seed window 1')

ax_bin1 = fig.add_subplot(gs[2,0])
nlp.plot_stat_map(ni1[0], axes=ax_bin1, colorbar=False, display_mode='x', cut_coords=(0,), cmap=cm.RdBu_r)
ax_bin1.set_title('winner take all partition')

ax_corr2 = fig.add_subplot(gs[1,1])
nlp.plot_glass_brain(ci2[0], cmap=hotcold, axes=ax_corr2, vmin=-1, vmax=1)
ax_corr2.set_title('seed window 2')

ax_bin2 = fig.add_subplot(gs[2,1])
nlp.plot_stat_map(ni2[0], axes=ax_bin2, colorbar=False, display_mode='x', cut_coords=(0,), cmap=cm.RdBu_r)
ax_bin2.set_title('winner take all partition')

ax_corr3 = fig.add_subplot(gs[1,2])
nlp.plot_glass_brain(ci3[0], cmap=hotcold, axes=ax_corr3, vmin=-1, vmax=1)
ax_corr3.set_title('seed window 3')

ax_bin3 = fig.add_subplot(gs[2,2])
nlp.plot_stat_map(ni3[0], axes=ax_bin3, colorbar=False, display_mode='x', cut_coords=(0,), cmap=cm.RdBu_r)
ax_bin3.set_title('winner take all partition')
fig.savefig('static_figure.svg', dpi=300)

