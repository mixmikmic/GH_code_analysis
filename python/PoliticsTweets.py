get_ipython().magic('matplotlib inline')

import GPy
from GPy.plotting.matplot_dep.util import fixed_inputs
from chained_gp.svgp_multi import SVGPMulti
from chained_gp.het_beta import HetBeta
from chained_gp.svgp_beta import SVGPBeta

import numpy as np

import scipy as sp
from scipy.stats import beta as beta_dist
from scipy import io

import pods
import pandas as pd

import pylab
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import num2date
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors
from matplotlib.dates import date2num
from matplotlib import rc
from matplotlib2tikz import save as tikz_save

import climin
import sys
from functools import partial

pylab.rcParams['figure.figsize'] = 10, 8  # that's default image size for this 
np.random.seed(0)

import time
now = time.time()
politics_dict = pods.datasets.politics_twitter()
download_time = time.time() - now
print download_time

ukip_df = politics_dict['ukip'].reset_index(drop=True)
greens_df = politics_dict['greens'].reset_index(drop=True)
conservative_df = politics_dict['conservative'].reset_index(drop=True)
labour_df = politics_dict['labour'].reset_index(drop=True)

#Merge all the dataframes
ukip_df['party'] = 'ukip'
greens_df['party'] = 'greens'
conservative_df['party'] = 'conservative'
labour_df['party'] = 'labour'

ukip_df['color'] = 'm'
greens_df['color'] = 'g'
conservative_df['color'] = 'b'
labour_df['color'] = 'r'

tweets_df = pd.concat([ukip_df, greens_df, conservative_df, labour_df])

tweets_df[tweets_df['party'] == 'ukip'].reset_index().describe()


tweets_df.ix[tweets_df['pos'] == 0.5, 'pos'] = np.nan
tweets_df.dropna(subset=['pos'], inplace=True)
tweets_df['time'] = pd.to_datetime(tweets_df['time'])
#Make timestamp from time object
tweets_df['timestamp'] = tweets_df['time'].apply(date2num)
parties_df = tweets_df.groupby('party')

parties = tweets_df['party'].unique()
num_parties = parties.shape[0]

labour_df.describe()

tweets_df[tweets_df['party'] == 'ukip']['time'].describe()

tweets_df[tweets_df['party'] == 'labour']['time'].describe()

fig, axes = plt.subplots(len(parties), 1, sharex=True)
fig_hist, hist_ax = plt.subplots(1,1)
for (party, party_df), ax in zip(parties_df, axes):
    print party
    color = party_df['color'][0]
    
    party_df['pos'].plot(kind='hist', alpha=0.49, normed=True, label=party, ax=hist_ax, color=color)
    plt.title("Distribution of tweets positiveness")

    ax.set_title("{} positiveness over time".format(party))
    party_df.plot(x='time', y='pos', ax=ax, c=color, label=party, legend=False, alpha=0.7)
    ax.set_ylim(0,1)
    ax.set_xlim(tweets_df['timestamp'].min(), tweets_df['timestamp'].max())
plt.legend()

fig, axes = plt.subplots(len(parties), 1, sharex=True)
for (party, party_df), ax in zip(parties_df, axes):
    color = party_df['color'][0]
    ax.set_title("{} positiveness over time".format(party))
    ax.plot_date(party_df['timestamp'], party_df['pos'], 'o', color=color, alpha=0.01, lw=0)

#Sort based on 'pos' with all parties together
transformed_tweets = tweets_df.sort(columns='pos')

#draw N random uniform 'pos' values
#replace each pos value with its corresponding (index) random value
transformed_tweets['pos'] = np.sort(np.random.uniform(0, 1, transformed_tweets.shape[0]))

transformed_grouped = transformed_tweets.groupby('party')

fig, axes = plt.subplots(len(parties), 1, sharex=True)
fig_hist, hist_ax = plt.subplots(1,1)
for (party, party_df), ax in zip(transformed_grouped, axes):
    print party
    color = party_df['color'][0]
    
    party_df['pos'].plot(kind='hist', alpha=0.49, normed=True, label=party, ax=hist_ax, color=color)
    plt.title("Distribution of transformed tweets positiveness")

    ax.set_title("{} positiveness over time".format(party))
    party_df.plot(x='time', y='pos', ax=ax, c=color, label=party, legend=False, alpha=0.7)
    ax.set_ylim(0,1)
    ax.set_xlim(tweets_df['timestamp'].min(), tweets_df['timestamp'].max())
plt.legend()

fig, axes = plt.subplots(len(parties), 1, sharex=True)
for (party, party_df), ax in zip(transformed_grouped, axes):
    color = party_df['color'][0]
    ax.set_title("{} positiveness over time".format(party))
    ax.plot_date(party_df['timestamp'], party_df['pos'], 'o', color=color, alpha=0.01, lw=0)

party = 'labour'
party_df = parties_df.get_group(party)
#party_df = transformed_grouped.get_group(party)  # The reweighted will produce more extreme effects
party_df = party_df.sort(columns='pos')
party_df = party_df.reindex(np.random.permutation(party_df.index))
color = party_df['color'][0]

#subsample = 5000
#if subsample is not None:
    #df = df.loc[np.random.choice(df.index, subsample, replace=False)]

X = party_df['timestamp'][:, None]
X_offset = X.mean()
X_scale = X.max() - X.min()
X = (X - X_offset)/X_scale
Z = np.linspace(X.min(), X.max(), 100)[:, None]
Y = party_df['pos'][:, None]

likelihood = HetBeta()

kernf = GPy.kern.Matern32(1, lengthscale=0.3, name='kernf_rbf1')
kernf += GPy.kern.White(1, variance=1e-5, name='f_white')
#kernf += GPy.kern.RBF(1, lengthscale=0.6, name='kernf_rbf2')
#Needs white or variance doesn't checkgrad!
kerng = GPy.kern.Matern32(1, lengthscale=0.3, name='kerng_rbf1')
kerng += GPy.kern.White(1, variance=1e-5, name='g_white')
#kerng += GPy.kern.RBF(1, lengthscale=0.6, name='kerng_rbf2')
kernf.name = 'kernf'
kerng.name = 'kerng'

kern = [kernf, kerng]

m = SVGPBeta(X, Y, Z, kern, likelihood, batchsize=500)

m.kernf.f_white.fix()
m.kerng.g_white.fix()
m.kernf.fix()
m.kerng.fix()
m.Z.fix()


opt = climin.Adadelta(m.optimizer_array, m.stochastic_grad, step_rate=0.1)
#opt = climin.RmsProp(m.optimizer_array, m.stochastic_grad, step_rate=0.04)

def callback(i, max_iter=5000):
    ll = m.log_likelihood()
    print str(i['n_iter']) + " " + str(ll) + " " + str(np.max(i['gradient'])), "\r",
    if np.isnan(ll):
        raise ValueError('Log likelihood went to nan')
    #Stop after max_iter iterations
    if i['n_iter'] > max_iter:
        return True
    return False

c_init = partial(callback, max_iter=1000)
info = opt.minimize_until(c_init)

# First 
m.kernf.constrain_positive()
m.kerng.constrain_positive()
opt = climin.Adadelta(m.optimizer_array, m.stochastic_grad, step_rate=0.03)
c_full = partial(callback, max_iter=500)
info = opt.minimize_until(c_full)

# Very simple annealing of step_rate
opt = climin.Adadelta(m.optimizer_array, m.stochastic_grad, step_rate=0.01)
c_full = partial(callback, max_iter=100)
info = opt.minimize_until(c_full)
opt = climin.Adadelta(m.optimizer_array, m.stochastic_grad, step_rate=0.001)
c_full = partial(callback, max_iter=100)
info = opt.minimize_until(c_full)
opt = climin.Adadelta(m.optimizer_array, m.stochastic_grad, step_rate=0.0001)
c_full = partial(callback, max_iter=100)
info = opt.minimize_until(c_full)


def plot_fs(self, dim=0, variances=False, median=True, true_variance=True,
                y_alpha=0.3, cmap=plt.cm.YlOrRd, num_pred_points=200,
                X_scale=1.0, X_offset=0.0, plot_dates=True, subsample=True):
    """
    Plotting for models with two latent functions, one is an exponent over the scale
    parameter
    """
    assert self.likelihood.request_num_latent_functions(self.Y) == 2
    if median:
        XX = fixed_inputs(self, non_fixed_inputs=[dim], fix_routine='median', as_list=False, X_all=True)
    else:
        XX = fixed_inputs(self, non_fixed_inputs=[dim], fix_routine='mean', as_list=False, X_all=True)
    #Now we have the other values fixed, remake the matrix to be the right shape
    XX = np.zeros((num_pred_points, self.X_all.shape[1]))
    for d in range(self.X_all.shape[1]):
        XX[:, d] = self.X_all[0, d]
    X_pred_points = XX.copy()
    X_pred_points_lin = np.linspace(self.X_all[:, dim].min(), self.X_all[:, dim].max(), XX.shape[0])
    X_pred_points[:, dim] = X_pred_points_lin

    mf, covf = self._raw_predict(X_pred_points, 0, full_cov=True)
    mg, covg = self._raw_predict(X_pred_points, 1, full_cov=True)

    covf = covf[:,:,0]
    covg = covg[:,:,0]

    num_samples = 60
    samples_f = np.random.multivariate_normal(mf.flatten(), covf, num_samples)
    samples_g = np.random.multivariate_normal(mg.flatten(), covg, num_samples)

    alpha = np.exp(samples_f)
    beta = np.exp(samples_g)

    num_y_pixels = 60
    #Want the top left pixel to be evaluated at 1
    line = np.linspace(1, 0, num_y_pixels)
    res = np.zeros((X_pred_points.shape[0], num_y_pixels))
    for j in range(X_pred_points.shape[0]):
        sf = alpha[:, j]  # Pick out the jth point along X axis
        sg = beta[:, j]
        for i in range(num_samples):
            # Pick out the sample and evaluate the pdf on a line between 0
            # and 1 with these alpha and beta values
            res[j, :] += beta_dist.pdf(line, sf[i], sg[i])
        res[j, :] /= num_samples

    vmax, vmin = res[np.isfinite(res)].max(), res[np.isfinite(res)].min()
    

    norm = matplotlib.colors.Normalize(vmax=vmax, vmin=vmin)
    
    X_all = self.X_all*X_scale + X_offset
    X_pred_points = X_pred_points*X_scale + X_offset
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)
    plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=2.0)
    ax1.set_title('averaged pdf and data')
    im = ax1.imshow(res.T, origin='upper', 
                    extent=[X_pred_points[:,dim].min(),X_pred_points[:,dim].max(), 0, 1],
                    aspect='auto', cmap=cmap, norm=norm)
    fig.colorbar(im, orientation='horizontal', pad=0.2)
    #Subsample and change y_alpha accordingly
    subsample_inds = np.random.permutation(range(X_all.shape[0]))[:int(X_all.shape[0]*subsample)]
    X_sub = X_all[subsample_inds, :]
    Y_sub = self.Y_all[subsample_inds, :]
    y_alpha = y_alpha/float(subsample)
    if plot_dates:
        #All others should follow suit since we sharex
        ax1.plot_date(X_sub, Y_sub, 'kx', alpha=y_alpha)
    else:
        ax1.plot(X_sub, Y_sub, 'kx', alpha=y_alpha)

    #For labels
    ax2.set_title('Posterior GP for Beta distributed variables')
    ax2.plot(X_pred_points, beta.T[:,0], 'b-', label='beta', alpha=3./num_samples)
    ax2.plot(X_pred_points, alpha.T[:,0], 'm-', label='alpha', alpha=3./num_samples)
    
    #For rest of samples
    ax2.plot(X_pred_points, beta.T[:,1:], 'b-', alpha=3./num_samples)
    ax2.plot(X_pred_points, alpha.T[:,1:], 'm-', alpha=3./num_samples)
    ax2.legend()
    
    ax3.plot(X_pred_points, alpha.T / (alpha.T + beta.T), 'b-', alpha=3./num_samples)
    ax3.set_title('Mean')

    var = (alpha.T*beta.T) / ((alpha.T + beta.T)**2 * (alpha.T+beta.T +1))
    ax4.plot(X_pred_points, var, 'b-', alpha=3./num_samples)
    ax4.set_title('variance')

    for i in range(num_samples):
        a = alpha[i, :]
        b = beta[i, :]
        mode = (a - 1) / (a + b - 2)
        mode = np.where(mode < 0, np.nan, mode)
        ax5.plot(X_pred_points, mode, 'b-', alpha=3./num_samples)
    ax5.set_title('Modes where they exist (alpha > 1, beta > 1)')
    ax5.set_ylim(0,1)
    plt.legend()

    ax1.set_xlim(X_pred_points[:, dim].min(), X_pred_points[:, dim].max())

    fig3d = plt.figure(figsize=(13,5))
    ax = fig3d.add_subplot(111, projection='3d')
    ax.view_init(elev=55., azim=300.0)
    axlim_min, axlim_max = X_pred_points[:, dim].min(), X_pred_points[:, dim].max()
    x, y = np.mgrid[axlim_min:axlim_max:complex(res.shape[0]),
                    1:0:complex(res.shape[1])]
    #x_dates = num2date(x)
    xfmt = mdates.DateFormatter('%b %d')
    ax.plot_surface(x,y,res,cmap=cmap,rstride=1, cstride=1, lw=0.05, alpha=1, edgecolor='b', norm=norm)
    #ax.xaxis.set_major_formatter(xfmt)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.set_zlabel('PDF')
    ax.set_ylabel('Sentiment')
    ax.set_xlabel('Date')
    #ax.autofmt_xdate()
    
    return fig, fig3d


m.plot_fs1 = partial(plot_fs, m)

m.plot_fs1(X_scale=X_scale, X_offset=X_offset, y_alpha=0.01)

m

import seaborn as sns
palette = sns.color_palette()

palette[0]

def rgb2hex(rgb):
    def clamp(x): 
        return max(0, min(int(x), 255))
    return "#{0:02x}{1:02x}{2:02x}".format(clamp(rgb[0]), clamp(rgb[1]), clamp(rgb[2]))

rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'figure.autolayout': True})

def save_plot_fs(self, dim=0, variances=False, median=True, true_variance=True,
                y_alpha=0.3, cmap=plt.cm.YlOrRd, num_pred_points=200,
                X_scale=1.0, X_offset=0.0, plot_dates=True, subsample=1.0):
    """
    Plotting for models with two latent functions, one is an exponent over the scale
    parameter
    """
    import seaborn as sns
    #sns.set_style(style='white')
    #palette = sns.color_palette("hls")
    assert self.likelihood.request_num_latent_functions(self.Y) == 2
    subsample = float(subsample)
    if median:
        XX = fixed_inputs(self, non_fixed_inputs=[dim], fix_routine='median', as_list=False, X_all=True)
    else:
        XX = fixed_inputs(self, non_fixed_inputs=[dim], fix_routine='mean', as_list=False, X_all=True)
    #Now we have the other values fixed, remake the matrix to be the right shape
    XX = np.zeros((num_pred_points, self.X_all.shape[1]))
    for d in range(self.X_all.shape[1]):
        XX[:, d] = self.X_all[0, d]
    X_pred_points = XX.copy()
    X_pred_points_lin = np.linspace(self.X_all[:, dim].min(), self.X_all[:, dim].max(), XX.shape[0])
    X_pred_points[:, dim] = X_pred_points_lin

    mf, covf = self._raw_predict(X_pred_points, 0, full_cov=True)
    mg, covg = self._raw_predict(X_pred_points, 1, full_cov=True)

    covf = covf[:,:,0]
    covg = covg[:,:,0]

    num_samples = 30
    samples_f = np.random.multivariate_normal(mf.flatten(), covf, num_samples)
    samples_g = np.random.multivariate_normal(mg.flatten(), covg, num_samples)

    alpha = np.exp(samples_f)
    beta = np.exp(samples_g)

    num_y_pixels = 40
    #Want the top left pixel to be evaluated at 1
    line = np.linspace(1, 0, num_y_pixels)
    res = np.zeros((X_pred_points.shape[0], num_y_pixels))
    for j in range(X_pred_points.shape[0]):
        sf = alpha[:, j]  # Pick out the jth point along X axis
        sg = beta[:, j]
        for i in range(num_samples):
            # Pick out the sample and evaluate the pdf on a line between 0
            # and 1 with these alpha and beta values
            res[j, :] += beta_dist.pdf(line, sf[i], sg[i])
        res[j, :] /= num_samples

    vmax, vmin = res[np.isfinite(res)].max(), res[np.isfinite(res)].min()
    

    norm = matplotlib.colors.Normalize(vmax=vmax, vmin=vmin)
    
    X_all = self.X_all*X_scale + X_offset
    X_pred_points = X_pred_points*X_scale + X_offset
    fig_data, ax1 = plt.subplots(1)
    fig_latents, ax2 = plt.subplots(1)
    fig_mean, ax3 = plt.subplots(1)
    fig_var, ax4 = plt.subplots(1)
    fig_modes, ax5 = plt.subplots(1)
    fig_mean_var, ax6 = plt.subplots(1)
    fig3d = plt.figure(figsize=(8,4))
    
    fig_data.set_figwidth(12)
    fig_data.set_figheight(2)
    fig_latents.set_figwidth(5)
    fig_latents.set_figheight(2)
    fig_mean.set_figwidth(5)
    fig_mean.set_figheight(2)
    fig_var.set_figwidth(5)
    fig_var.set_figheight(2)
    fig_modes.set_figwidth(5)
    fig_modes.set_figheight(2)
    fig_mean_var.set_figwidth(5)
    fig_mean_var.set_figheight(2)
    
    #ax1.set_title('Beta PDF')
    im = ax1.imshow(res.T, origin='upper', 
                    extent=[X_pred_points[:,dim].min(),X_pred_points[:,dim].max(), 0, 1],
                    aspect='auto', cmap=cmap, norm=norm)
    fig.colorbar(im, orientation='horizontal', pad=0.2)
    #Subsample and change y_alpha accordingly
    subsample_inds = np.random.permutation(range(X_all.shape[0]))[:int(X_all.shape[0]*subsample)]
    X_sub = X_all[subsample_inds, :]
    Y_sub = self.Y_all[subsample_inds, :]
    y_alpha = y_alpha/float(subsample)
    if plot_dates:
        #All others should follow suit since we sharex
        ax1.plot_date(X_sub, Y_sub, 'kx', alpha=y_alpha)
    else:
        ax1.plot(X_sub, Y_sub, 'kx', alpha=y_alpha)
    ax1.set_ylabel('Sentiment')
    
    #For labels
    ax2.set_title(r'Posterior GP for latent functions of $\textrm{Beta}(\alpha,\beta)$')
    ax2.plot(X_pred_points, beta.T[:,0], 'b-', label=r'$\beta$', alpha=3./num_samples)
    ax2.plot(X_pred_points, alpha.T[:,0], 'm-', label=r'$\alpha$', alpha=3./num_samples)
    
    #For rest of samples
    ax2.plot(X_pred_points, beta.T[:,1:], 'b-', alpha=3./num_samples)
    ax2.plot(X_pred_points, alpha.T[:,1:], 'm-', alpha=3./num_samples)

    
    ax3.plot(X_pred_points, alpha.T / (alpha.T + beta.T), 'b-', alpha=3./num_samples)
    ax3.set_title('Mean')

    var = (alpha.T*beta.T) / ((alpha.T + beta.T)**2 * (alpha.T+beta.T +1))
    ax4.plot(X_pred_points, var, 'b-', alpha=3./num_samples)
    ax4.set_title('Variance')

    for i in range(num_samples):
        a = alpha[i, :]
        b = beta[i, :]
        mode = (a - 1) / (a + b - 2)
        mode = np.where(mode < 0, np.nan, mode)
        ax5.plot(X_pred_points, mode, 'b-', alpha=3./num_samples)
    ax5.set_title(r'Modes where they exist ($\alpha > 1$, $\beta > 1$)', fontsize=10)
    ax5.set_ylim(0,1)


    ax = fig3d.add_subplot(111, projection='3d')
    ax.view_init(elev=55., azim=300.0)
    axlim_min, axlim_max = X_pred_points[:, dim].min(), X_pred_points[:, dim].max()
    x, y = np.mgrid[axlim_min:axlim_max:complex(res.shape[0]),
                    1:0:complex(res.shape[1])]
    #x_dates = num2date(x)
    xfmt = mdates.DateFormatter('%b %d')
    xfmt = mdates.DateFormatter('%m/%d/%y')
    p = ax.plot_surface(x,y,res,cmap=cmap,rstride=1, cstride=1, lw=0.01, alpha=1, edgecolor='b', norm=norm)
    ax.xaxis.set_major_formatter(xfmt)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.set_zlabel('PDF')
    ax.set_ylabel('Sentiment')

    #plt.colorbar(p, orientation='vertical', pad=0.1)
    
    #Twin plot the mean and variance
    ax6.plot(X_pred_points, alpha.T / (alpha.T + beta.T), 'b-', alpha=3./num_samples)
    for tl in ax6.get_yticklabels():
        tl.set_color('b')
    ax6.set_ylabel('Mean', color='b')
    ax6.set_title('Mean and Variance of Beta($\alpha$, $\beta$)')
    
    ax7 = ax6.twinx()
    var = (alpha.T*beta.T) / ((alpha.T + beta.T)**2 * (alpha.T+beta.T +1))
    ax7.plot(X_pred_points, var, 'm-', alpha=3./num_samples)
    for tl in ax7.get_yticklabels():
        tl.set_color('m')
    ax7.set_ylabel('Variance', color='m')
    ax6.grid(True)
    
    ax2.legend(loc='lower right', bbox_to_anchor=(1.15, 0.7))
    ax1.set_xlim(X_pred_points[:, dim].min(), X_pred_points[:, dim].max())
    ax2.set_xlim(X_pred_points[:, dim].min(), X_pred_points[:, dim].max())
    ax3.set_xlim(X_pred_points[:, dim].min(), X_pred_points[:, dim].max())
    ax4.set_xlim(X_pred_points[:, dim].min(), X_pred_points[:, dim].max())
    ax5.set_xlim(X_pred_points[:, dim].min(), X_pred_points[:, dim].max())
    ax6.set_xlim(X_pred_points[:, dim].min(), X_pred_points[:, dim].max())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
    ax1.xaxis.set_major_locator(mdates.DayLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
    ax3.xaxis.set_major_locator(mdates.DayLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
    ax4.xaxis.set_major_locator(mdates.DayLocator())
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
    ax5.xaxis.set_major_locator(mdates.DayLocator())
    ax6.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
    ax6.xaxis.set_major_locator(mdates.DayLocator())
    ax2.locator_params(axis='y', nbins=5)
    ax3.locator_params(axis='y', nbins=5)
    ax4.locator_params(axis='y', nbins=5)
    ax5.locator_params(axis='y', nbins=5)
    ax6.locator_params(axis='y', nbins=5)
    ax7.locator_params(axis='y', nbins=4)
    ax.locator_params(axis='z', nbins=5)
    #Makes me want to cry a little bit inside.
    ax.zaxis._axinfo['label']['space_factor'] = 2.0
    ax.yaxis._axinfo['label']['space_factor'] = 2.0
    #fig_data.autofmt_xdate()
    fig_latents.autofmt_xdate()
    fig_mean.autofmt_xdate()
    fig_var.autofmt_xdate()
    fig_modes.autofmt_xdate()
    fig_mean_var.autofmt_xdate()
    fig3d.autofmt_xdate()
    fig3d.tight_layout()
    ax.autoscale(tight=True)
    return fig_data, fig_latents, fig_mean, fig_var, fig_modes, fig3d, fig_mean_var

from functools import partial
m.save_plot_fs = partial(save_plot_fs, m)

m.save_plot_fs(X_scale=X_scale, X_offset=X_offset, y_alpha=0.01, subsample=0.1)
#fig_data, fig_latents, fig_mean, fig_var, fig_modes, fig3d = m.save_plot_fs(X_scale=X_scale, X_offset=X_offset, y_alpha=0.01, subsample=0.1)

save = False
if save:
    #Save all formats
    fig_data.savefig('labour_pdf.eps', rasterized=True, dpi=100, bbox_inches='tight')
    fig_data.savefig('labour_pdf.pdf', rasterized=True, dpi=100, bbox_inches='tight')

    fig3d.savefig('labour_3d.eps', rasterized=True, dpi=100, bbox_inches='tight', pad_inches=1.0)
    fig3d.savefig('labour_3d.pdf', rasterized=True, dpi=100, bbox_inches='tight', pad_inches=1.0)

    fig_latents.savefig('labour_latent.eps', rasterized=True, dpi=100, bbox_inches='tight')
    fig_latents.savefig('labour_latent.pdf', rasterized=True, dpi=100, bbox_inches='tight')

    tikz_save('labour_latent.tikz', fig_latents)
    tikz_save('labour_mean.tikz', fig_mean)
    tikz_save('labour_var.tikz', fig_var)
    tikz_save('labour_modes.tikz', fig_modes)
    tikz_save('labour_mean_vars.tikz', fig_mean_vars)
    #tikz_save('labour_pdf.tikz', fig_data)
    #tikz_save('labour_3d.tikz', fig3d)

m



