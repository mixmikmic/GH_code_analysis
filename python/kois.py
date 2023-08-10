get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpld3

from gprot.summary import summarize_fits

mcq_df = pd.read_csv('../gprot/data/mcquillan_acf_kois.txt', comment='#')
mcq_df.index = mcq_df.koi_id
df = summarize_fits('../kois/results') # Requires data.
df.koi = pd.Series(df.index).apply(lambda s: int(s[4:]))
df.index = df.koi
print('Fit results of {} KOIs loaded.'.format(len(df)))

import corner
cols = ['ln_A_50', 'ln_l_50', 'ln_G_50', 'ln_sigma_50', 'ln_period_50']
mpld3.disable_notebook()
corner.corner(df[cols], labels=cols);

subdf = df.ix[mcq_df.ix[df.koi].dropna().index].join(mcq_df) # is this using the right indices?
subdf['mcq_ln_period'] = np.log(subdf.P_rot)
subdf[['ln_period_16', 'ln_period_50', 'ln_period_84','mcq_ln_period']].head()
print('{} in common with McQ+13.'.format(len(subdf)))

mpld3.enable_notebook()

def scatter_compare(df):
    fig, ax = plt.subplots(1,1)
    points = ax.scatter(df.mcq_ln_period, df.ln_period_50, alpha=0.2)
    ax.plot(range(6), range(6), 'k', lw=2, alpha=0.5, zorder=0)
    ax.plot(range(6), range(6) + np.log(2), 'k', ls='--', lw=2, alpha=0.2, zorder=0)
    ax.plot(range(6), range(6) - np.log(2), 'k', ls='--', lw=2, alpha=0.2, zorder=0)
    ax.set_xlim((0,5))
    ax.set_ylim((0,5))
    tooltip = mpld3.plugins.PointLabelTooltip(points, labels=['{}: {:.2f} ({:.2f})'.format(i, df.ix[i, 'P_rot'],
                                                                                     np.exp(df.ix[i, 'ln_period_50']))
                                                        for i in df.index])
    mpld3.plugins.connect(fig, tooltip)
    return fig

scatter_compare(subdf);

y_int = 6.5
Glim = 0
def good_mask(df, has_truth=True, max_unc=0.2, Glim=0., y_int=y_int):
    if has_truth:
        is_close = (np.absolute(df.mcq_ln_period - df.ln_period_50) < 0.1)
    else:
        is_close = np.ones(len(df), dtype=bool)
        
    return (is_close &
        ((df.ln_period_84 - df.ln_period_16) < max_unc) &
        (df.ln_G_50 > Glim) & 
        (df.ln_l_50 < (df.ln_G_50 + y_int)))

mpld3.enable_notebook()
def plot_Gl(df, good, alpha=0.3):
    x = 'ln_G_50'
    y = 'ln_l_50'
    bad = ~good
    plt.scatter(df[good][x], df[good][y], color='b', alpha=alpha)
    plt.scatter(df[bad][x], df[bad][y], color='r', alpha=alpha);
    x = np.linspace(Glim, 4, 100)
    y = x + y_int
    plt.plot(x, y, 'k:')
    plt.plot([Glim, Glim], [0, y_int-Glim], 'k:')
    plt.ylim(ymin=1)
    
good = good_mask(subdf, has_truth=False, max_unc=0.5)
bad = ~good
print('{} classified as good; {} bad.'.format(good.sum(), bad.sum()))
plot_Gl(subdf, good);

scatter_compare(subdf[good]);

good_all = good_mask(df, has_truth=False, max_unc=0.3)
good_all.sum()

plot_Gl(df, good_all);

df_sample = df.join(mcq_df, how='inner')
good_sample = good_mask(df_sample, has_truth=False, max_unc=0.3)
n_mcq = len(df_sample.dropna())
print('{} good Prots identified out of {}.  McQ got {}'.format(good_sample.sum(),
                                                              len(df_sample),
                                                              n_mcq))
plot_Gl(df, good_all);

