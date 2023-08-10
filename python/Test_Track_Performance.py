import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import pickle
from scipy.stats import ks_2samp

plt.style.use('seaborn-poster')
plt.rcParams['figure.figsize'] = 12,15

def draw_boxes(ax, data1, data2):
    
    meanpointprops = dict(marker='.', markeredgecolor='black',
                      markerfacecolor='black')

    bplot = ax.boxplot(data1, showmeans=True, meanprops=meanpointprops, vert=False)

    ax.plot(data2, range(1,49), '.', color='grey', markersize=8)

    plt.setp(bplot['medians'], color='black')

    [[item.set_color('black') for item in bplot['means']]]
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    return ax

def draw_labels(ax, ssr_x, crsr_x, xlab, df):
    for i in range(0, 51, 8):

        ax.axhspan((i-4)+.5, i+.5, color='k', alpha=0.1)

    ax.plot([-150, 150], [12.5, 12.5], color='k',linewidth=2)
    ax.plot([-150, 150], [24.5, 24.5], color='k',linewidth=2)
    ax.plot([-150, 150], [36.5, 36.5], color='k',linewidth=2)

    CRSR = df['CRSR'].values
    SSR = df['SSR'].values
    P = df['MCS_proba'].values

    for i in range(len(CRSR)):
        ax.annotate("CRSR: " +str(CRSR[i]).zfill(2) + " km",
                     xy=(crsr_x, i+.75), fontsize=12)

    SSR = [192, 96, 48] * 4

    for n, s in enumerate(SSR):

        plt.annotate("SSR: " + str(s) + " km", xy=(ssr_x, (n*4)+2.5), fontsize=15)

    locs = [6, 18, 30, 42]

    labs = ["P$_{mcs}$=\n" + l for l in ['0.95', '0.90', '0.50', '0.00']]
    
    ax.set_xlabel(xlab)
    ax.set_yticks(locs)
    ax.set_yticklabels(labs)
    ax.grid(which='major', linestyle='-', linewidth='0.15', color='black',zorder=0)
    plt.gca().yaxis.grid(False)   
    return ax

def load_data(year, var_name):
    
    dr = "../data/track_data/sum_stats/" + str(year) + "/" + str(year) + "_"

    df_un = pickle.load(open(dr + var_name + "_unmatched_master.pkl", 'rb'))
    df_re = pickle.load(open(dr + var_name + "_rematched_master.pkl", 'rb'))

    df_un = df_un.sort_values(by=['MCS_proba', 'SSR', 'CRSR'], ascending=False)
    df_re = df_re.sort_values(by=['MCS_proba', 'SSR', 'CRSR'], ascending=False)
    
    return df_un, df_re

def sig_diffs(unmatched, rematched):
    
    data1 = unmatched['Distribution'].values
    data2 = rematched['Distribution'].values
    
    crsr = rematched['CRSR'].values
    ssr = rematched['SSR'].values
    prob = rematched['MCS_proba'].values
    
    d = {'CRSR':[], 'SSR':[], 'MCS_proba':[], 'ks_stat':[], 'p_val':[], 'Significant': []}
    
    for idx, (x1, x2) in enumerate(zip(data1, data2)):
    
        ks_s, p = ks_2samp(x1, x2)

        d['CRSR'].append(crsr[idx])
        d['SSR'].append(ssr[idx])
        d['MCS_proba'].append(prob[idx])
        d['ks_stat'].append(ks_s)
        d['p_val'].append(p)
        
        if p < 0.001:
            d['Significant'].append("YES")
        if p >= 0.001:
            d['Significant'].append("--")
    
    return d

df_un, df_re = load_data(2015, 'mean_dur')

fig, ax = plt.subplots(1)

ax = draw_boxes(ax, df_re['Distribution'].values, df_un['mean'].values)

ax = draw_labels(ax, 13.5, 16.6, "Swath Duration (hrs)", df_re)

ax.set_xlim(0, 19)
ax.set_ylim(.5, 48.5)

max_diff = np.max(df_re['mean'].values - df_un['mean'].values)
min_diff = np.min(df_re['mean'].values - df_un['mean'].values)
max_loc = np.argmax(df_re['mean'].values - df_un['mean'].values)
min_loc = np.argmin(df_re['mean'].values - df_un['mean'].values)

print('Max Diff (hrs):', max_diff,
      'CRSR:', df_re['CRSR'].values[max_loc], 
      'SSR:', df_re['SSR'].values[max_loc],
      'MCS_Proba:', df_re['MCS_proba'].values[max_loc])

print('Min Diff (hrs):', min_diff,
      'CRSR:', df_re['CRSR'].values[min_loc], 
      'SSR:', df_re['SSR'].values[min_loc],
      'MCS_Proba:', df_re['MCS_proba'].values[min_loc])

sig_test = pd.DataFrame.from_dict(sig_diffs(df_un, df_re))
sig_test = sig_test.sort_values(by=['MCS_proba', 'SSR', 'CRSR'], ascending=True)

print(sig_test[['CRSR', 'SSR', 'MCS_proba', 'ks_stat', 'p_val', 'Significant']])

df_un, df_re = load_data(2016, 'mean_dur')

fig, ax = plt.subplots(1)

ax = draw_boxes(ax, df_re['Distribution'].values, df_un['mean'].values)

ax = draw_labels(ax, 13.5, 16.6, "Swath Duration (hrs)", df_re)

ax.set_xlim(0, 19)
ax.set_ylim(.5, 48.5)

max_diff = np.max(df_re['mean'].values - df_un['mean'].values)
min_diff = np.min(df_re['mean'].values - df_un['mean'].values)
max_loc = np.argmax(df_re['mean'].values - df_un['mean'].values)
min_loc = np.argmin(df_re['mean'].values - df_un['mean'].values)

print('Max Diff (hrs):', max_diff,
      'CRSR:', df_re['CRSR'].values[max_loc], 
      'SSR:', df_re['SSR'].values[max_loc],
      'MCS_Proba:', df_re['MCS_proba'].values[max_loc])

print('Min Diff (hrs):', min_diff,
      'CRSR:', df_re['CRSR'].values[min_loc], 
      'SSR:', df_re['SSR'].values[min_loc],
      'MCS_Proba:', df_re['MCS_proba'].values[min_loc])

sig_test = pd.DataFrame.from_dict(sig_diffs(df_un, df_re))
sig_test = sig_test.sort_values(by=['MCS_proba', 'SSR', 'CRSR'], ascending=True)

print(sig_test[['CRSR', 'SSR', 'MCS_proba', 'ks_stat', 'p_val', 'Significant']])

df_un, df_re = load_data(2015, 'std_refl')

fig, ax = plt.subplots(1)

ax = draw_boxes(ax, df_re['Distribution'].values, df_un['mean'].values)

ax = draw_labels(ax, 12.1, 13.75, "Intensity Error (dBZ)", df_re)

ax.set_xlim(5, 15)
ax.set_ylim(.5, 48.5)

max_diff = np.max(df_re['mean'].values - df_un['mean'].values)
min_diff = np.min(df_re['mean'].values - df_un['mean'].values)
max_loc = np.argmax(df_re['mean'].values - df_un['mean'].values)
min_loc = np.argmin(df_re['mean'].values - df_un['mean'].values)

print('Max Diff (hrs):', max_diff,
      'CRSR:', df_re['CRSR'].values[max_loc], 
      'SSR:', df_re['SSR'].values[max_loc],
      'MCS_Proba:', df_re['MCS_proba'].values[max_loc])

print('Min Diff (hrs):', min_diff,
      'CRSR:', df_re['CRSR'].values[min_loc], 
      'SSR:', df_re['SSR'].values[min_loc],
      'MCS_Proba:', df_re['MCS_proba'].values[min_loc])

sig_test = pd.DataFrame.from_dict(sig_diffs(df_un, df_re))
sig_test = sig_test.sort_values(by=['MCS_proba', 'SSR', 'CRSR'], ascending=True)

print(sig_test[['CRSR', 'SSR', 'MCS_proba', 'ks_stat', 'p_val', 'Significant']])

df_un, df_re = load_data(2016, 'std_refl')

fig, ax = plt.subplots(1)

ax = draw_boxes(ax, df_re['Distribution'].values, df_un['mean'].values)

ax = draw_labels(ax, 12.1, 13.75, "Intensity Error (dBZ)", df_re)

ax.set_xlim(5, 15)
ax.set_ylim(.5, 48.5)

max_diff = np.max(df_re['mean'].values - df_un['mean'].values)
min_diff = np.min(df_re['mean'].values - df_un['mean'].values)
max_loc = np.argmax(df_re['mean'].values - df_un['mean'].values)
min_loc = np.argmin(df_re['mean'].values - df_un['mean'].values)

print('Max Diff (hrs):', max_diff,
      'CRSR:', df_re['CRSR'].values[max_loc], 
      'SSR:', df_re['SSR'].values[max_loc],
      'MCS_Proba:', df_re['MCS_proba'].values[max_loc])

print('Min Diff (hrs):', min_diff,
      'CRSR:', df_re['CRSR'].values[min_loc], 
      'SSR:', df_re['SSR'].values[min_loc],
      'MCS_Proba:', df_re['MCS_proba'].values[min_loc])

sig_test = pd.DataFrame.from_dict(sig_diffs(df_un, df_re))
sig_test = sig_test.sort_values(by=['MCS_proba', 'SSR', 'CRSR'], ascending=True)

print(sig_test[['CRSR', 'SSR', 'MCS_proba', 'ks_stat', 'p_val', 'Significant']])

df_un, df_re = load_data(2015, 'lin_err')

fig, ax = plt.subplots(1)

ax = draw_boxes(ax, df_re['Distribution'].values, df_un['mean'].values)

ax = draw_labels(ax, 95, 115, "Linearity Error (km)", df_re)

ax.set_xlim(0, 131)
ax.set_ylim(.5, 48.5)

max_diff = np.max(df_re['mean'].values - df_un['mean'].values)
min_diff = np.min(df_re['mean'].values - df_un['mean'].values)
max_loc = np.argmax(df_re['mean'].values - df_un['mean'].values)
min_loc = np.argmin(df_re['mean'].values - df_un['mean'].values)

print('Max Diff (hrs):', max_diff,
      'CRSR:', df_re['CRSR'].values[max_loc], 
      'SSR:', df_re['SSR'].values[max_loc],
      'MCS_Proba:', df_re['MCS_proba'].values[max_loc])

print('Min Diff (hrs):', min_diff,
      'CRSR:', df_re['CRSR'].values[min_loc], 
      'SSR:', df_re['SSR'].values[min_loc],
      'MCS_Proba:', df_re['MCS_proba'].values[min_loc])

sig_test = pd.DataFrame.from_dict(sig_diffs(df_un, df_re))
sig_test = sig_test.sort_values(by=['MCS_proba', 'SSR', 'CRSR'], ascending=True)

print(sig_test[['CRSR', 'SSR', 'MCS_proba', 'ks_stat', 'p_val', 'Significant']])

df_un, df_re = load_data(2016, 'lin_err')

fig, ax = plt.subplots(1)

ax = draw_boxes(ax, df_re['Distribution'].values, df_un['mean'].values)

ax = draw_labels(ax, 95, 115, "Linearity Error (km)", df_re)

ax.set_xlim(0, 131)
ax.set_ylim(.5, 48.5)

max_diff = np.max(df_re['mean'].values - df_un['mean'].values)
min_diff = np.min(df_re['mean'].values - df_un['mean'].values)
max_loc = np.argmax(df_re['mean'].values - df_un['mean'].values)
min_loc = np.argmin(df_re['mean'].values - df_un['mean'].values)

print('Max Diff (hrs):', max_diff,
      'CRSR:', df_re['CRSR'].values[max_loc], 
      'SSR:', df_re['SSR'].values[max_loc],
      'MCS_Proba:', df_re['MCS_proba'].values[max_loc])

print('Min Diff (hrs):', min_diff,
      'CRSR:', df_re['CRSR'].values[min_loc], 
      'SSR:', df_re['SSR'].values[min_loc],
      'MCS_Proba:', df_re['MCS_proba'].values[min_loc])

sig_test = pd.DataFrame.from_dict(sig_diffs(df_un, df_re))
sig_test = sig_test.sort_values(by=['MCS_proba', 'SSR', 'CRSR'], ascending=True)

print(sig_test[['CRSR', 'SSR', 'MCS_proba', 'ks_stat', 'p_val', 'Significant']])

def get_normalized(df1, df2, df3):
    
    data1 = df1['mean'].values
    data2 = df2['mean'].values
    data3 = df3['mean'].values
    
    return ((data1/np.max(data1))+(data2/np.max(data2))-(data3/np.max(data3)))

_, df_dur = load_data(2015, 'mean_dur')
_, df_std = load_data(2015, 'std_refl')
_, df_lin = load_data(2015, 'lin_err')

vals = get_normalized(df_std, df_lin, df_dur)

fig, ax = plt.subplots(1)

ax.plot(vals - np.mean(vals), list(range(1,49)), 'k.')

ax = draw_labels(ax, .11, .17, "Normalized Error (Difference From Mean)", df_dur)

plt.plot([0, 0], [0, 50], 'k--',linewidth=2)
ax.set_xlim(-.2, .22)
ax.set_ylim(.5, 48.5)

_, df_dur = load_data(2016, 'mean_dur')
_, df_std = load_data(2016, 'std_refl')
_, df_lin = load_data(2016, 'lin_err')

vals = get_normalized(df_std, df_lin, df_dur)

fig, ax = plt.subplots(1)

ax.plot(vals - np.mean(vals), list(range(1,49)), 'k.')

ax = draw_labels(ax, .11, .17, "Normalized Error (Difference From Mean)", df_dur)

plt.plot([0, 0], [0, 50], 'k--',linewidth=2)
ax.set_xlim(-.2, .22)
ax.set_ylim(.5, 48.5)

