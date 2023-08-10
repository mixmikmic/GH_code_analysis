import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

pp_fn = '/project/meteo/scratch/S.Rasp/convective_variability_data/preproc_data/variability/all_days_m_50_mems_inc_1h.nc'

rg = nc.Dataset(pp_fn)

# Extract the fields
mean_m = rg.variables['mean_m'][:]
var_m = rg.variables['var_m'][:]
mean_N = rg.variables['mean_N'][:]
mean_M = rg.variables['mean_M'][:]

beta = var_m / (mean_m ** 2)

print(rg.variables['date'][:])
print(rg.variables['time'][:])


def plot_m_fields(day_ind, time_ind, mean_m, beta, mean_N):
    # Loop over different scales
    for i, i_n in enumerate([6, 3, 0]):
        n = rg.variables['n'][i_n]
        print('Coarsening scale = %0.1f km' % (n * 2.8))

        lim = int(256 / n)
        plot_m = mean_m[day_ind, time_ind, i_n, :lim, :lim]
        plot_beta = beta[day_ind, time_ind, i_n, :lim, :lim]
        plot_N = mean_N[day_ind, time_ind, i_n, :lim, :lim]

        inds = (np.isfinite(plot_beta) * plot_N*50 > 10)
        print('beta = %.2f; beta (only more than 10 clouds) = %.2f; beta (10+ and N weighted) = %.2f' % 
              (np.nanmean(plot_beta), np.average(plot_beta[inds]), np.average(plot_beta[inds], weights=plot_N[inds]))) 

        fig, axarr = plt.subplots(1, 3, figsize=(17, 5))

        im = axarr[0].imshow(plot_m, cmap='rainbow', origin='lower')
        fig.colorbar(im, ax=axarr[0], shrink=0.8)
        axarr[0].set_title('mean(m)')


        im = axarr[1].imshow(plot_beta, cmap='bwr', origin='lower', vmin=0, vmax=2)
        fig.colorbar(im, ax=axarr[1], shrink=0.8)
        axarr[1].set_title('beta')

        im = axarr[2].imshow(plot_N * 50, cmap='rainbow', origin='lower')
        fig.colorbar(im, ax=axarr[2], shrink=0.8)
        axarr[2].set_title('Number of clouds in all 50 mems')
        plt.show()

plot_m_fields(7, 10, mean_m, beta, mean_N)

plot_m_fields(1, 15, mean_m, beta, mean_N)

for i_n, n in enumerate(rg.variables['n']):
    all_m = np.ravel(mean_m[:, :, i_n])[np.isfinite(np.ravel(mean_m[:, :, i_n]))]
    all_M = np.ravel(mean_M[:, :, i_n])[np.isfinite(np.ravel(mean_M[:, :, i_n]))]
    print('Scale = %0.1f km; Correlation coefficient = %0.2f' % 
          ((n * 2.8), np.corrcoef(all_m, all_M)[0, 1]))
    plt.scatter(all_M, all_m)
    plt.xlabel('M')
    plt.ylabel('m')
    plt.show()

from IPython.display import Image
Image(filename = '/home/s/S.Rasp/repositories/convective_variability_analysis/figures/cloud_stats/m_evolution_all-days_50-mems_perim-3_dr-2_var-m_inc-1h_sum.png', width=500, height=500)

flat_mean_m = np.ravel(mean_m[:, :, 6])[np.isfinite(np.ravel(mean_m[:, :, 6]))]
_ = plt.hist(flat_mean_m, bins = 100)

pp_fn2 = '/project/meteo/scratch/S.Rasp/convective_variability_data/preproc_data/variability/hist_full.nc'
rg2 = nc.Dataset(pp_fn2)

cond_m_hist = rg2.variables['cond_m_hist'][:]

for i_n, n in enumerate(rg2.variables['n']):
    print(str(n * 2.8))
    fig, ax = plt.subplots(1, 1)
    for j, m in enumerate(rg2.variables['cond_bins_mean_m']):
        hist_data = cond_m_hist[i_n, j]
        hist_data /= np.sum(hist_data)
        ax.plot(rg2.variables['cond_bins_m'], hist_data, label=str(m))
        ax.set_yscale('log')
    plt.legend()
    plt.show()



