import pandas as pd
import numpy as np
import h5py
import os

model_folder_name = '13_cccfff'
main_path = '/fhgfs/users/jbehnken/01_Data/04_Models'
prediction_path = os.path.join(main_path, model_folder_name, model_folder_name.split('_')[-1]+'_prediction.csv')
plotting_path = os.path.join(main_path, model_folder_name, model_folder_name.split('_')[-1]+'_plotting.h5')

model_folder_name = '16_pre-cccfff'
prediction_path = '/fhgfs/users/jbehnken/01_Data/04_Models/16_pre-cccfff/models_folder/103_with_5_100_pre_flat/Prediction.csv'
plotting_path = '/fhgfs/users/jbehnken/01_Data/04_Models/16_pre-cccfff/models_folder/103_with_5_100_pre_flat/Plotting.h5'

keys_1 = ['event_num', 'night', 'run_id', 'theta', 'theta_deg', 'theta_deg_off_1', 'theta_deg_off_2', 'theta_deg_off_3', 'theta_deg_off_4', 'theta_deg_off_5', 'theta_off_1', 'theta_off_1_rec_pos', 'theta_off_2', 'theta_off_2_rec_pos', 'theta_off_3', 'theta_off_3_rec_pos', 'theta_off_4', 'theta_off_4_rec_pos', 'theta_off_5', 'theta_off_5_rec_pos', 'theta_rec_pos', 'unix_time_utc']
keys_2 = ['run_start', 'run_stop', 'source']

with h5py.File('/net/big-tank/POOL/projects/fact/datasets/Crab1314_darknight_std_analysis_0.17.2.hdf5', 'r') as f:
    data_1  = []
    data_2  = []
    for key in keys_1:
        data_1.append(np.array(f['events'][key]))
    for key in keys_2:
        data_2.append(np.array(f['runs'][key]))
        
data_1 = list(map(list, zip(*data_1)))
df_crab = pd.DataFrame(data_1)
df_crab.columns = keys_1
del data_1

data_2 = list(map(list, zip(*data_2)))
df_crab_runs = pd.DataFrame(data_2)
df_crab_runs.columns = keys_2
del data_2

df_crab.shape

df_crab_runs.head()

df_pred = pd.read_csv(prediction_path)
df_pred.columns = ['night', 'run_id', 'event_num', 'proton_prediction', 'gamma_prediction']
df_pred.head(2)

df_merged = pd.merge(df_pred, df_crab, how='inner', on=['event_num', 'night', 'run_id'])
df_merged.shape

with h5py.File(plotting_path, 'w') as hdf:
    events = hdf.create_group('events')
    dset_gamma_prediction = events.create_dataset('gamma_prediction', data=df_merged['gamma_prediction'].values)
    dset_theta = events.create_dataset('theta', data=df_merged['theta'].values)
    dset_theta_deg_off_1 = events.create_dataset('theta_deg_off_1', data=df_merged['theta_deg_off_1'].values)
    dset_theta_deg_off_2 = events.create_dataset('theta_deg_off_2', data=df_merged['theta_deg_off_2'].values)
    dset_theta_deg_off_3 = events.create_dataset('theta_deg_off_3', data=df_merged['theta_deg_off_3'].values)
    dset_theta_deg_off_4 = events.create_dataset('theta_deg_off_4', data=df_merged['theta_deg_off_4'].values)
    dset_theta_deg_off_5 = events.create_dataset('theta_deg_off_5', data=df_merged['theta_deg_off_5'].values)
    dset_theta_deg = events.create_dataset('theta_deg', data=df_merged['theta_deg'].values)
    dset_unix_time_utc = events.create_dataset('unix_time_utc', data=df_merged['unix_time_utc'].values.tolist())
    
    runs = hdf.create_group('runs')
    dset_run_start = runs.create_dataset('run_start', data=df_crab_runs['run_start'].values.tolist())
    dset_run_stop = runs.create_dataset('run_stop', data=df_crab_runs['run_stop'].values.tolist())



def theta_square_plot(theta2_cut = 0.8, data_path = plotting_path, key = 'events', start = None, end = None, threshold = 0.5, bins = 40, alpha = 0.2, output = False):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import h5py
    from dateutil.parser import parse as parse_date

    from fact.io import read_h5py
    from fact.analysis import (li_ma_significance, split_on_off_source_dependent,)
    import click

    columns = [
        'gamma_prediction',
        'theta_deg',
        'theta_deg_off_1',
        'theta_deg_off_2',
        'theta_deg_off_3',
        'theta_deg_off_4',
        'theta_deg_off_5',
        'unix_time_utc',
    ]

    stats_box_template = r'''Source: {source}, $t_\mathrm{{obs}} = {t_obs:.2f}\,\mathrm{{h}}$
    $N_\mathrm{{On}} = {n_on}$, $N_\mathrm{{Off}} = {n_off}$, $\alpha = {alpha}$
    $N_\mathrm{{Exc}} = {n_excess:.1f} \pm {n_excess_err:.1f}$, $S_\mathrm{{Li&Ma}} = {significance:.1f}\,\sigma$
    '''


    theta_cut = np.sqrt(theta2_cut)

    with h5py.File(data_path, 'r') as f:
        source_dependent = 'gamma_prediction_off_1' in f[key].keys()

    if source_dependent:
        print('Separation was using source dependent features')
        columns.extend('gamma_prediction_off_' + str(i) for i in range(1, 6))
        theta_cut = np.inf
        theta2_cut = np.inf

    events = read_h5py(data_path, key='events', columns=columns)
    events['timestamp'] = pd.to_datetime(events['unix_time_utc_0'] * 1e6 + events['unix_time_utc_1'],unit='us',)
    runs = read_h5py(data_path, key='runs')
    runs['run_start'] = pd.to_datetime(runs['run_start'])
    runs['run_stop'] = pd.to_datetime(runs['run_stop'])

    if start is not None:
        events = events.query('timestamp >= @start')
        runs = runs.query('run_start >= @start')
    if end is not None:
        events = events.query('timestamp <= @end')
        runs = runs.query('run_stop <= @end')

    if source_dependent:
        on_data, off_data = split_on_off_source_dependent(events, threshold)
        theta_on = on_data.theta_deg
        theta_off = off_data.theta_deg
    else:
        selected = events.query('gamma_prediction >= {}'.format(threshold))
        theta_on = selected.theta_deg
        theta_off = pd.concat([selected['theta_deg_off_{}'.format(i)]for i in range(1, 6)])

    del events

    if source_dependent:
        limits = [0,max(np.percentile(theta_on, 99)**2, np.percentile(theta_off, 99)**2),]
    else:
        limits = [0, 0.3]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    h_on, bin_edges = np.histogram(theta_on.apply(lambda x: x**2).values,bins=bins,range=limits)
    h_off, bin_edges, _ = ax.hist(theta_off.apply(lambda x: x**2).values,bins=bin_edges,range=limits,weights=np.full(len(theta_off), 0.2),histtype='stepfilled',color='lightgray',)

    bin_center = bin_edges[1:] - np.diff(bin_edges) * 0.5
    bin_width = np.diff(bin_edges)

    ax.errorbar(bin_center,h_on,yerr=np.sqrt(h_on) / 2,xerr=bin_width / 2,linestyle='',label='On',)
    ax.errorbar(bin_center,h_off,yerr=alpha * np.sqrt(h_off) / 2,xerr=bin_width / 2,linestyle='',label='Off',)

    if not source_dependent:
        ax.axvline(theta_cut**2, color='gray', linestyle='--')

    n_on = np.sum(theta_on < theta_cut)
    n_off = np.sum(theta_off < theta_cut)
    significance = li_ma_significance(n_on, n_off, alpha=alpha)


    ax.text(0.5, 0.95,stats_box_template.format(source='Crab',t_obs=0,n_on=n_on, n_off=n_off, alpha=alpha,n_excess=n_on - alpha * n_off,n_excess_err=np.sqrt(n_on + alpha**2 * n_off),significance=significance,),transform=ax.transAxes,fontsize=12,va='top',ha='center',)

    ax.set_xlabel(r'$(\theta / {}^\circ )^2$')
    ax.legend()
    fig.tight_layout()
    plt.xlim(0.0, 0.3)

    if output:
        fig.savefig(output, dpi=300)
    else:
        plt.show()

for i in np.arange(0.2, 0.3, 0.01):
    print(i)
    theta_square_plot(theta2_cut=0.8, data_path=plotting_path, key='events', start=None, end=None, threshold=i, bins=60, alpha=0.2, output=False)



theta_square_plot(theta2_cut=0.8, data_path=plotting_path, key='events', start=None, end=None, threshold=0.05, bins=60, alpha=0.2, output=False)



