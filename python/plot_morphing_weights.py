import numpy as np
import sys
import math
from scipy.stats import norm

get_ipython().magic('matplotlib inline')
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.mlab import griddata

sys.path.append('..')
from higgs_inference import settings

data_dir = '../data'
figure_dir = '../figures'

thetas = settings.thetas
theta1 = settings.theta1_default

xi = np.linspace(-1.0, 1.0, settings.n_thetas_roam)
yi = np.linspace(-1.0, 1.0, settings.n_thetas_roam)
xx, yy = np.meshgrid(xi, yi)
thetas_eval = np.asarray(list(zip(xx.flatten(), yy.flatten())))

r_eval = np.load('../results/truth/r_roam_truth.npy')[0]
r_thetas = np.load('../results/truth/r_roam_thetas_truth.npy')[:,0]

margin_l_absolute = 8. * 0.1
margin_r_absolute = 8. * 0.06
margin_sep_absolute = 8. * 0.02
margin_t_absolute = 8. * 0.02
margin_b_absolute = 8. * 0.08

def calculate_height(n_panels=2, width=8., panel_aspect_ratio=1.):
    
    # Calculate horizontal margins. Units: relative to width.
    margin_l = margin_l_absolute / width
    margin_r = margin_r_absolute / width
    margin_l_subsequent = margin_l
    if n_panels > 2:
        margin_l_subsequent = margin_r
    margin_sep = margin_sep_absolute / width
    if n_panels > 2:
        margin_sep = 0
    margin_sep_total = margin_r + margin_sep + margin_l_subsequent
    panel_width = (1. - margin_l - margin_r - (n_panels - 1)*margin_sep_total) / n_panels
    
    # Calculate wspace argument of subplots_adjust
    wspace = margin_sep_total / panel_width
    
    # Calculate absolute height
    panel_height_absolute = panel_width * width / panel_aspect_ratio
    
    # Calculate horizontal margins. Units: relative to width.
    height = panel_height_absolute + margin_t_absolute + margin_b_absolute
    panel_height = panel_height_absolute / height
    margin_t = margin_t_absolute / height
    margin_b = margin_b_absolute / height
    
    # Return height
    return height


def adjust_margins(n_panels=2, width=8., panel_aspect_ratio=1.):
    
    # Calculate horizontal margins. Units: relative to width.
    margin_l = margin_l_absolute / width
    margin_r = margin_r_absolute / width
    margin_l_subsequent = margin_l
    if n_panels > 2:
        margin_l_subsequent = margin_r
    margin_sep = margin_sep_absolute / width
    if n_panels > 2:
        margin_sep = 0
    margin_sep_total = margin_r + margin_sep + margin_l_subsequent
    panel_width = (1. - margin_l - margin_r - (n_panels - 1)*margin_sep_total) / n_panels
    
    # Calculate wspace argument of subplots_adjust
    wspace = margin_sep_total / panel_width
    
    # Calculate absolute height
    panel_height_absolute = panel_width * width / panel_aspect_ratio
    
    # Calculate horizontal margins. Units: relative to width.
    height = panel_height_absolute + margin_t_absolute + margin_b_absolute
    panel_height = panel_height_absolute / height
    margin_t = margin_t_absolute / height
    margin_b = margin_b_absolute / height
    
    # Set margins
    plt.subplots_adjust(left = margin_l,
                        right = 1. - margin_r,
                        bottom = margin_b,
                        top = 1. - margin_t,
                        wspace = wspace)
    
print(calculate_height(2,8.))
print(calculate_height(3,9.))

# Morphing
n_samples = 15

def calculate_wtilde(t, component_sample):
    wtilde_components = np.asarray([
        1. + 0. * t[:, 0],
        t[:, 1],
        t[:, 1] * t[:, 1],
        t[:, 1] * t[:, 1] * t[:, 1],
        t[:, 1] * t[:, 1] * t[:, 1] * t[:, 1],
        t[:, 0],
        t[:, 0] * t[:, 1],
        t[:, 0] * t[:, 1] * t[:, 1],
        t[:, 0] * t[:, 1] * t[:, 1] * t[:, 1],
        t[:, 0] * t[:, 0],
        t[:, 0] * t[:, 0] * t[:, 1],
        t[:, 0] * t[:, 0] * t[:, 1] * t[:, 1],
        t[:, 0] * t[:, 0] * t[:, 0],
        t[:, 0] * t[:, 0] * t[:, 0] * t[:, 1],
        t[:, 0] * t[:, 0] * t[:, 0] * t[:, 0]
    ]).T
    return wtilde_components.dot(component_sample)


def calculate_wi(t, component_sample, sigma_sample, sigma_component):
    wtildes = calculate_wtilde(t, component_sample)
    sigma_wtildes = sigma_sample * wtildes # (?, 15)
    
    denom = np.ones_like(sigma_wtildes.T) # (?, 15)
    denom[:,:] /= np.sum(sigma_wtildes, axis=1)
    denom = denom.T
    
    return (sigma_wtildes * denom).T

thetas_morphing = np.array([
    [0.,0.],
    [0.,0.25],
    [0.,0.5],
    [0.,0.75],
    [0.,1.],
    [0.25,0.],
    [0.25,0.25],
    [0.25,0.5],
    [0.25,0.75],
    [0.5,0.],
    [0.5,0.25],
    [0.5,0.5],
    [0.75,0.],
    [0.75,0.25],
    [1.,0.]
])

sigma_component = np.load(data_dir + '/morphing/component_xsec.npy')[1:] # Ignore background component
component_sample = np.load(data_dir + '/morphing/component_sample.npy')[1:] # Ignore background component
sigma_sample = np.linalg.inv(component_sample).dot(sigma_component)

wi_original = calculate_wi(thetas_eval, component_sample, sigma_sample, sigma_component)

wi_original = wi_original.reshape(15,settings.n_thetas_roam,settings.n_thetas_roam)

wi_original_sum = np.sum(wi_original, axis=0)
print(np.min(wi_original_sum), np.max(wi_original_sum))
print(np.var(wi_original))

wi_original = np.clip(wi_original,-1000.,1000.)

fig = plt.figure(figsize=(15,20))

for i in range(15):
    ax = plt.subplot(5, 3, i+1)
    
    pcm = ax.pcolormesh(xi, yi, wi_original[i],
                       norm=matplotlib.colors.SymLogNorm(linthresh=0.1, linscale=1.,
                                              vmin=-1000.0, vmax=1000.0),
                       cmap='PRGn')
    cbar = fig.colorbar(pcm, ax=ax, extend='both')

    plt.scatter(thetas_morphing[:, 0], thetas_morphing[:, 1],
                marker='o', c='black', s=25, lw=0, zorder=9, alpha=0.3)
    plt.scatter([thetas_morphing[i, 0]], [thetas_morphing[i, 1]],
                marker='o', c='black', s=50, lw=0, zorder=9, alpha=1.)

    plt.xlim(-1., 1)
    plt.ylim(-1., 1.)
    plt.xlabel(r"$f_{W} \, v^2 / \Lambda^2$")
    plt.ylabel(r"$f_{WW} \, v^2 / \Lambda^2$")
    cbar.set_label(r'$w_{'+ str(i) + r'}$')

plt.tight_layout()
plt.show()

th_morphing = [0, 101, 106, 902, 910,
              226, 373, 583, 747, 841,
              599, 709, 422, 367, 167]

benchmarks = [708, 9, 422]

plt.figure(figsize=(12,12))

for i, theta in enumerate(thetas):
    plt.text(theta[0],theta[1],str(i),
             fontsize = 14 if i in benchmarks else 10,
             alpha = 1. if i in th_morphing or i in benchmarks or i in settings.extended_pbp_training_thetas else 0.5,
             color = 'purple' if i in th_morphing and i in benchmarks else 'red' if i in th_morphing else 'blue' if i in benchmarks else 'black' if i in settings.extended_pbp_training_thetas else 'grey',
             horizontalalignment = 'center',
             verticalalignment = 'center')
    
plt.xlim(-1.05,1.05)
plt.ylim(-1.05,1.05)

plt.tight_layout()

thetas_used_morphing = thetas[th_morphing]

sample_component = np.load(data_dir + '/morphing/components_fakebasis2.npy')[:,1:] # Ignore background component
component_sample = np.linalg.inv(sample_component)
sigma_sample = np.load(data_dir + '/morphing/fakebasis2_xsecs.npy')
sigma_component = component_sample.dot(sigma_sample)

wi_used = calculate_wi(thetas_eval, component_sample, sigma_sample, sigma_component)
wi_used = wi_used.reshape(15,settings.n_thetas_roam,settings.n_thetas_roam)
wi_theta1_used = calculate_wi(np.array([thetas[theta1]]), component_sample, sigma_sample, sigma_component).flatten()

wi_used_sum = np.sum(wi_used, axis=0)
print(np.min(wi_used_sum), np.max(wi_used_sum))
print(np.var(wi_used))

wi_used = np.clip(wi_used,-1000.,1000.)
print(wi_used.shape)

fig = plt.figure(figsize=(15,20))

for i in range(15):
    ax = plt.subplot(5, 3, i+1)
    
    pcm = ax.pcolormesh(xi, yi, wi_used[i],
                       norm=matplotlib.colors.SymLogNorm(linthresh=0.1, linscale=1.,
                                              vmin=-100.0, vmax=100.0),
                       cmap='PRGn')
    cbar = fig.colorbar(pcm, ax=ax, extend='both')

    plt.scatter(thetas_used_morphing[:, 0], thetas_used_morphing[:, 1],
                marker='o', c='black', s=25, lw=0, zorder=9, alpha=0.3)
    plt.scatter([thetas_used_morphing[i, 0]], [thetas_used_morphing[i, 1]],
                marker='o', c='black', s=50, lw=0, zorder=9, alpha=1.)

    plt.xlim(-1.0, 1.0)
    plt.ylim(-1.0, 1.0)
    plt.xlabel(r"$f_{W} \, v^2 / \Lambda^2$")
    plt.ylabel(r"$f_{WW} \, v^2 / \Lambda^2$")
    cbar.set_label(r'$w_{'+ str(i) + r'}$')

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(9.,calculate_height(2,9.,panel_aspect_ratio=1.2)))

for i in range(2):
    ax = plt.subplot(1, 2, i+1)
    
    pcm = ax.pcolormesh(xi, yi, wi_used[i],
                       norm=matplotlib.colors.SymLogNorm(linthresh=0.1, linscale=1.,
                                              vmin=-100.0, vmax=100.0),
                       cmap='PRGn')
    cbar = fig.colorbar(pcm, ax=ax, extend='both')

    plt.scatter(thetas_used_morphing[:, 0], thetas_used_morphing[:, 1],
                marker='o', c='0.4', s=35, lw=0, zorder=9, alpha=1.)
    plt.scatter([thetas_used_morphing[i, 0]], [thetas_used_morphing[i, 1]],
                marker='o', c='black', s=70, lw=0, zorder=9, alpha=1.)

    plt.xlim(-1.0, 1.0)
    plt.ylim(-1.0, 1.0)
    plt.xlabel(r"$f_{W} \, v^2 / \Lambda^2$")
    plt.ylabel(r"$f_{WW} \, v^2 / \Lambda^2$")
    cbar.set_label(r'$w_{'+ str(i) + r'}$')

adjust_margins(2,9.,panel_aspect_ratio=1.2)
plt.savefig(figure_dir + "/morphing.pdf")

def _clip(x, clip=1.e-6):
    y = np.sign(x)*np.clip(np.abs(x),1.e-6,None)
    y[(y<clip) & (y > -clip)] = clip
    return y
    

uncertainty_on_log_r_from_morphing = np.zeros((101, 101))

for i, (wi, thi) in enumerate(zip(wi_used, th_morphing)):
    uncertainty_on_log_r_from_morphing += wi**2 * r_thetas[thi]**2
    
uncertainty_on_log_r_from_morphing = np.sqrt(uncertainty_on_log_r_from_morphing)
uncertainty_on_log_r_from_morphing /= r_eval.reshape((101,101))

print(np.min(uncertainty_on_log_r_from_morphing), np.median(uncertainty_on_log_r_from_morphing), np.max(uncertainty_on_log_r_from_morphing))

uncertainty_on_log_r_from_decomposition = np.zeros((101, 101))

for i, (wi, wi1, thi) in enumerate(zip(wi_used, wi_theta1_used, th_morphing)):
    for j, (wj, wj1, thj) in enumerate(zip(wi_used[:i], wi_theta1_used[:i], th_morphing[:i])):
    
        denominator1 = np.zeros((101, 101))
        for k, (wk1, thk) in enumerate(zip(wi_theta1_used[:], th_morphing[:])):
            denominator1 += wk1 / _clip(wi) * wk1 * r_thetas[thk] / r_thetas[thi]
        denominator1 = denominator1**2
        
        term1 = wj1 / _clip(wi) * r_thetas[thj] / r_thetas[thi] / denominator1
        
        denominator2 = np.zeros((101, 101))
        for k, (wk1, thk) in enumerate(zip(wi_theta1_used[:], th_morphing[:])):
            denominator2 += wk1 / _clip(wj) * wk1 * r_thetas[thk] / r_thetas[thj]
        denominator2 = denominator2**2
        
        term2 = wi1 / _clip(wj) * r_thetas[thi] / r_thetas[thj] / denominator2
        
        uncertainty_on_log_r_from_decomposition += (term1 - term2)**2
    
uncertainty_on_log_r_from_decomposition = np.sqrt(uncertainty_on_log_r_from_decomposition)
uncertainty_on_log_r_from_decomposition /= r_eval.reshape((101,101))

print(np.min(uncertainty_on_log_r_from_decomposition), np.median(uncertainty_on_log_r_from_decomposition), np.max(uncertainty_on_log_r_from_decomposition))

fig = plt.figure(figsize=(10.,4.))

ax = plt.subplot(1,2,1)
pcm = ax.pcolormesh(xi, yi, uncertainty_on_log_r_from_morphing,
                   norm=matplotlib.colors.LogNorm(vmin=1., vmax=1.e2),
                   cmap='viridis_r')
cbar = fig.colorbar(pcm, ax=ax, extend='both')
plt.scatter(thetas_used_morphing[:, 0], thetas_used_morphing[:, 1],
            marker='o', c='white', s=50, lw=0, zorder=9, alpha=1.)
plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
plt.xlabel(r"$f_{W} \, v^2 / \Lambda^2$")
plt.ylabel(r"$f_{WW} \, v^2 / \Lambda^2$")
cbar.set_label(r'$\Delta \, \log \, r \; / \; \Delta \, \log \, r_i$ (numerator-only morphing)')

ax = plt.subplot(1,2,2)
pcm = ax.pcolormesh(xi, yi, uncertainty_on_log_r_from_decomposition,
                   norm=matplotlib.colors.LogNorm(vmin=1., vmax=1.e2),
                   cmap='viridis_r')
cbar = fig.colorbar(pcm, ax=ax, extend='both')
plt.scatter(thetas_used_morphing[:, 0], thetas_used_morphing[:, 1],
            marker='o', c='white', s=50, lw=0, zorder=9, alpha=1.)
plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
plt.xlabel(r"$f_{W} \, v^2 / \Lambda^2$")
plt.ylabel(r"$f_{WW} \, v^2 / \Lambda^2$")
cbar.set_label(r'$\Delta \, \log \, r \; / \; \Delta \, \log \, r_{i,j}$ (double decomposition)')

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(4.5,calculate_height(1,4.5,panel_aspect_ratio=1.2)))
ax = plt.gca()

pcm = ax.pcolormesh(xi, yi, uncertainty_on_log_r_from_morphing,
                   norm=matplotlib.colors.LogNorm(vmin=0.7, vmax=2e2),
                   cmap='viridis_r')
cbar = fig.colorbar(pcm, ax=ax, extend='both')
plt.scatter(thetas_used_morphing[:, 0], thetas_used_morphing[:, 1],
            marker='o', c='white', s=35, lw=0, zorder=9, alpha=1.)
plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
plt.xlabel(r"$f_{W} \, v^2 / \Lambda^2$")
plt.ylabel(r"$f_{WW} \, v^2 / \Lambda^2$")
cbar.set_label(r'$\Delta \, \log \, \hat{r} \, (x_e \, | \, \theta, \theta_1) \; / \; \Delta \, \log \, \hat{r}_c \, (x_e \, | \, \theta, \theta_1) $')

adjust_margins(1,4.5)
plt.savefig(figure_dir + "/morphing_uncertainties.pdf")

def add_noise(original, noise, relative=False):
    if relative:
        return norm(loc=original, scale=noise*np.abs(original)).rvs(size=original.shape)
    return norm(loc=original, scale=noise).rvs(size=original.shape)

def calculate_log_r_from_morphing(log_ri_noise=0., relative=False):
    
    log_ri = np.log(r_thetas[th_morphing])
    log_ri = add_noise(log_ri, log_ri_noise, relative)
    
    log_r = np.zeros_like(wi_used[0])
    for this_log_ri, this_wi in zip(log_ri, wi_used):
        log_r += this_wi*np.exp(this_log_ri)
    log_r = np.log(log_r)
        
    return log_r

def calculate_log_r_from_double_decomposition(log_ri_noise=0., log_rij_noise=0., relative=False):
    
    log_ri = np.log(r_thetas[th_morphing])
    log_ri = add_noise(log_ri, log_ri_noise, relative)
    log_rij = np.zeros((15,15))
    for i in range(15):
        for j in range(i):
            log_rij[i,j] = log_ri[i] - log_ri[j]
    log_rij = add_noise(log_rij, log_rij_noise, relative)
    for i in range(15):
        log_rij[i,i] = 0.
        for j in range(i):
            log_rij[j,i] = - log_rij[i,j]
    
    log_r = np.zeros_like(wi_used[0])
    for i, (wi, wi1) in enumerate(zip(wi_used, wi_theta1_used)):
        summand = np.zeros_like(wi)
        for j, (wj, wj1) in enumerate(zip(wi_used, wi_theta1_used)):
            summand += wj1 / _clip(wi) * np.exp(log_rij[j,i])
        log_r += 1./summand
        
    log_r = np.log(log_r)
    
        
    return log_r

noise = 0.001

log_r_from_morphing = []
log_r_from_decomposition = []
log_r_from_morphing_relative = []
log_r_from_decomposition_relative = []

for i in range(100):
    log_r_from_morphing.append(calculate_log_r_from_morphing(noise, False))
    log_r_from_decomposition.append(calculate_log_r_from_double_decomposition(0.,noise, False))
    log_r_from_morphing_relative.append(calculate_log_r_from_morphing(noise, True))
    log_r_from_decomposition_relative.append(calculate_log_r_from_double_decomposition(0.,noise, True))

log_r_from_morphing = np.asarray(log_r_from_morphing)
log_r_from_decomposition = np.asarray(log_r_from_decomposition)
log_r_from_morphing_relative = np.asarray(log_r_from_morphing_relative)
log_r_from_decomposition_relative = np.asarray(log_r_from_decomposition_relative)

median_log_r_from_morphing_relative = np.median(log_r_from_morphing_relative, axis=0)
median_log_r_from_decomposition_relative = np.median(log_r_from_decomposition_relative, axis=0)

uncertainty_on_log_r_from_morphing = np.sqrt(np.var(log_r_from_morphing, axis=0)) / noise
uncertainty_on_log_r_from_decomposition = np.sqrt(np.var(log_r_from_decomposition, axis=0)) / noise
uncertainty_on_log_r_from_morphing_relative = np.sqrt(np.var(log_r_from_morphing_relative, axis=0)) / noise / np.abs(median_log_r_from_morphing_relative)
uncertainty_on_log_r_from_decomposition_relative = np.sqrt(np.var(log_r_from_decomposition_relative, axis=0)) / noise / np.abs(median_log_r_from_decomposition_relative)

print(np.min(uncertainty_on_log_r_from_morphing), np.min(uncertainty_on_log_r_from_decomposition))
print(np.mean(uncertainty_on_log_r_from_morphing), np.mean(uncertainty_on_log_r_from_decomposition))
print(np.max(uncertainty_on_log_r_from_morphing), np.max(uncertainty_on_log_r_from_decomposition))

fig = plt.figure(figsize=(15.,4.))

ax = plt.subplot(1,3,1)
pcm = ax.pcolormesh(xi, yi, uncertainty_on_log_r_from_morphing,
                   norm=matplotlib.colors.LogNorm(vmin=1., vmax=100.),
                   cmap='viridis_r')
cbar = fig.colorbar(pcm, ax=ax, extend='both')
plt.scatter(thetas_used_morphing[:, 0], thetas_used_morphing[:, 1],
            marker='o', c='white', s=50, lw=0, zorder=9, alpha=1.)
plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
plt.xlabel(r"$f_{W} \, v^2 / \Lambda^2$")
plt.ylabel(r"$f_{WW} \, v^2 / \Lambda^2$")
cbar.set_label(r'$\Delta \, \log \, r \; / \; \Delta \, \log \, r_i$ (numerator-only morphing)')

ax = plt.subplot(1,3,2)
pcm = ax.pcolormesh(xi, yi, uncertainty_on_log_r_from_decomposition,
                   norm=matplotlib.colors.LogNorm(vmin=1., vmax=100.),
                   cmap='viridis_r')
cbar = fig.colorbar(pcm, ax=ax, extend='both')
plt.scatter(thetas_used_morphing[:, 0], thetas_used_morphing[:, 1],
            marker='o', c='white', s=50, lw=0, zorder=9, alpha=1.)
plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
plt.xlabel(r"$f_{W} \, v^2 / \Lambda^2$")
plt.ylabel(r"$f_{WW} \, v^2 / \Lambda^2$")
cbar.set_label(r'$\Delta \, \log \, r \; / \; \Delta \, \log \, r_{i,j}$ (double decomposition)')

ax = plt.subplot(1,3,3)
pcm = ax.pcolormesh(xi, yi, uncertainty_on_log_r_from_decomposition / uncertainty_on_log_r_from_morphing,
                   cmap='PRGn', vmin=0.7,vmax=1.3)
cbar = fig.colorbar(pcm, ax=ax, extend='both')
plt.scatter(thetas_used_morphing[:, 0], thetas_used_morphing[:, 1],
            marker='o', c='white', s=50, lw=0, zorder=9, alpha=1.)
plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
plt.xlabel(r"$f_{W} \, v^2 / \Lambda^2$")
plt.ylabel(r"$f_{WW} \, v^2 / \Lambda^2$")
cbar.set_label(r'Ratio: double decomposition error / morphing error')

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(15.,4.))

ax = plt.subplot(1,3,1)
pcm = ax.pcolormesh(xi, yi, uncertainty_on_log_r_from_morphing_relative,
                   norm=matplotlib.colors.LogNorm(vmin=1., vmax=1000.),
                   cmap='viridis_r')
cbar = fig.colorbar(pcm, ax=ax, extend='both')
plt.scatter(thetas_used_morphing[:, 0], thetas_used_morphing[:, 1],
            marker='o', c='white', s=50, lw=0, zorder=9, alpha=1.)
plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
plt.xlabel(r"$f_{W} \, v^2 / \Lambda^2$")
plt.ylabel(r"$f_{WW} \, v^2 / \Lambda^2$")
cbar.set_label(r'$\Delta \log r / |\log r| \; / \; \Delta \, \log \, r_i)/|\log r_i|$ (numerator-only morphing)')

ax = plt.subplot(1,3,2)
pcm = ax.pcolormesh(xi, yi, uncertainty_on_log_r_from_decomposition_relative,
                   norm=matplotlib.colors.LogNorm(vmin=1., vmax=1000.),
                   cmap='viridis_r')
cbar = fig.colorbar(pcm, ax=ax, extend='both')
plt.scatter(thetas_used_morphing[:, 0], thetas_used_morphing[:, 1],
            marker='o', c='white', s=50, lw=0, zorder=9, alpha=1.)
plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
plt.xlabel(r"$f_{W} \, v^2 / \Lambda^2$")
plt.ylabel(r"$f_{WW} \, v^2 / \Lambda^2$")
cbar.set_label(r'$\Delta \log r / |\log r| \; / \; \Delta \, \log \, r_{i,j})/|\log r_{i,j}|$ (double decomposition)')

ax = plt.subplot(1,3,3)
pcm = ax.pcolormesh(xi, yi, uncertainty_on_log_r_from_decomposition_relative / uncertainty_on_log_r_from_morphing_relative,
                   cmap='PRGn', vmin=0.5,vmax=1.5)
cbar = fig.colorbar(pcm, ax=ax, extend='both')
plt.scatter(thetas_used_morphing[:, 0], thetas_used_morphing[:, 1],
            marker='o', c='white', s=50, lw=0, zorder=9, alpha=1.)
plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
plt.xlabel(r"$f_{W} \, v^2 / \Lambda^2$")
plt.ylabel(r"$f_{WW} \, v^2 / \Lambda^2$")
cbar.set_label(r'Ratio: double decomposition error / morphing error')

plt.tight_layout()
plt.show()

ri_nottrained = np.load('../results/parameterized/morphing_wi_nottrained_carl_aware_basis.npy')
wi_nottrained = np.load('../results/parameterized/morphing_ri_nottrained_carl_aware_basis.npy')[0]
r_nottrained = np.load('../results/parameterized/r_nottrained_carl_aware_basis.npy')
ri_trained = np.load('../results/parameterized/morphing_wi_trained_carl_aware_basis.npy')
wi_trained = np.load('../results/parameterized/morphing_ri_trained_carl_aware_basis.npy')[0]
r_trained = np.load('../results/parameterized/r_trained_carl_aware_basis.npy')

theta_trained = 422
theta_nottrained = 9

wi_trained_true = calculate_wi(np.asarray([thetas[theta_trained]]),
                               component_sample, sigma_sample, sigma_component).reshape((-1,))
wi_nottrained_true = calculate_wi(np.asarray([thetas[theta_nottrained]]),
                               component_sample, sigma_sample, sigma_component).reshape((-1,))

print(wi_trained)
print(wi_trained_true)
print(wi_trained - wi_trained_true)

print('')

print(wi_nottrained)
print(wi_nottrained_true)
print(wi_nottrained-wi_nottrained_true)

plt.figure(figsize=(10.,5.))

plt.subplot(1,2,1)
plt.scatter(ri_trained[:,12], r_trained, s=15., alpha=0.3)

plt.subplot(1,2,2)
plt.scatter(ri_nottrained.dot(wi_nottrained), r_nottrained, s=15., alpha=0.3)

plt.show()



