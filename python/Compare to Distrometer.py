#Use Pyart35 env

import pyart
import statsmodels.api as sm

import netCDF4
import numpy as np
import platform
from matplotlib import pyplot as plt, rc
from glob import glob
import os
from datetime import datetime, timedelta
from scipy import interpolate, stats
import fnmatch
import matplotlib.dates as mdates
from pytmatrix.tmatrix import TMatrix, Scatterer
from pytmatrix.tmatrix_psd import TMatrixPSD, GammaPSD
from pytmatrix import orientation, radar, tmatrix_aux, refractive
import pickle
from pytmatrix.psd import PSDIntegrator, GammaPSD
from matplotlib.colors import LogNorm
#import pydisdrometer

get_ipython().magic('matplotlib inline')

#determine system

my_system = platform.system()
if my_system == 'Darwin':
    top = '/data/sample_sapr_data/sgpstage/sur/'
    s_dir = '/data/sample_sapr_data/sgpstage/interp_sonde/'
    d_dir = '/data/agu2016/dis/'
    odir_r = '/data/agu2016/radars/'
    odir_s = '/data/agu2016/stats/'
    odir_i = '/data/agu2016/images/'
    p_dir = '/data/agu2016/pars/'
elif my_system == 'Linux':
    top = '/lcrc/group/earthscience/radar/sgpstage/sur/'
    s_dir = '/lcrc/group/earthscience/radar/sgpstage/interp_sonde/'
    odir_r = '/lcrc/group/earthscience/radar/agu2016/radars/'
    odir_s = '/lcrc/group/earthscience/radar/agu2016/stats/'
    odir_i = '/lcrc/group/earthscience/radar/agu2016/images/'

#index all distrometer files

def get_file_tree(start_dir, pattern):
    """
    Make a list of all files matching pattern
    above start_dir

    Parameters
    ----------
    start_dir : string
        base_directory

    pattern : string
        pattern to match. Use * for wildcard

    Returns
    -------
    files : list
        list of strings
    """

    files = []

    for dir, _, _ in os.walk(start_dir):
        files.extend(glob(os.path.join(dir, pattern)))
    return files

def match_distro_file(distro_date, dis_file_list):
    #sgpvdisC1.b1.20110623.000000.cdf
    pattern = datetime.strftime(distro_date,
                        '*sgpvdisC1.b1.%Y%m%d.*')
    dis_name = fnmatch.filter(dis_file_list, pattern)[0]
    return dis_name

def masked_conv(string, bad_val = -9999.):
    try:
        out = float(string)
    except:
        out = bad_val
    
    return out

def read_dis(dfile):
    fh = open(dfile)
    line = fh.readline()
    fh.close()
    return line

# For testing variable aspect ratio
def drop_ar(D_eq):
    if D_eq < 0.7:
        return 1.0;
    elif D_eq < 1.5:
        return 1.173 - 0.5165*D_eq + 0.4698*D_eq**2 - 0.1317*D_eq**3 -             8.5e-3*D_eq**4
    else:
        return 1.065 - 6.25e-2*D_eq - 3.99e-3*D_eq**2 + 7.66e-4*D_eq**3 -             4.095e-5*D_eq**4 

def scatter_off_2dvd(d_diameters, d_densities):
    mypds = interpolate.interp1d(d_diameters,d_densities, bounds_error=False, fill_value=0.0)
    tm = TMatrixPSD(wavelength=tmatrix_aux.wl_C, m=refractive.m_w_10C[tmatrix_aux.wl_C])
    tm.psd_eps_func = lambda D: 1.0/drop_ar(D)
    tm.D_max = 10.0
    tm.or_pdf = orientation.gaussian_pdf(20.0)
    tm.orient = orientation.orient_averaged_fixed
    tm.geometries = (tmatrix_aux.geom_horiz_back, tmatrix_aux.geom_horiz_forw)
    tm.init_scatter_table()
    tm.psd = mypds#GammaPSD(D0=D0[i], Nw=Nw[j], mu=4)
    zdr=radar.Zdr(tm)
    z=radar.refl(tm)
    tm.set_geometry(tmatrix_aux.geom_horiz_forw)
    kdp=radar.Kdp(tm)
    A=radar.Ai(tm)
    return z,zdr,kdp,A

def scatter_off_2dvd_updated(d_diameters, d_densities):
    mypds = interpolate.interp1d(d_diameters,d_densities, bounds_error=False, fill_value=0.0)
    scatterer = Scatterer(wavelength=tmatrix_aux.wl_C, m=refractive.m_w_10C[tmatrix_aux.wl_C])
    #tm = TMatrixPSD(wavelength=tmatrix_aux.wl_C, m=refractive.m_w_10C[tmatrix_aux.wl_C])
    scatterer.psd_integrator = PSDIntegrator()
    scatterer.psd_integrator.axis_ratio_func = lambda D: 1.0/drop_ar(D)
    scatterer.psd_integrator.D_max = 10.0
    #tm.psd_eps_func = lambda D: 1.0/drop_ar(D)
    #tm.D_max = 10.0
    scatterer.psd_integrator.geometries = (tmatrix_aux.geom_horiz_back, tmatrix_aux.geom_horiz_forw)
    scatterer.or_pdf = orientation.gaussian_pdf(20.0)
    scatterer.orient = orientation.orient_averaged_fixed
    scatterer.psd_integrator.init_scatter_table(scatterer)
    scatterer.psd = mypds#GammaPSD(D0=2.0, Nw=1e3, mu=4)
    radar.refl(scatterer)
    #tm.or_pdf = orientation.gaussian_pdf(20.0)
    #tm.orient = orientation.orient_averaged_fixed
    #tm.geometries = (tmatrix_aux.geom_horiz_back, tmatrix_aux.geom_horiz_forw)
    #tm.init_scatter_table()
    #tm.psd = mypds#GammaPSD(D0=D0[i], Nw=Nw[j], mu=4)
    zdr=radar.Zdr(scatterer)
    z=radar.refl(scatterer)
    scatterer.set_geometry(tmatrix_aux.geom_horiz_forw)
    kdp=radar.Kdp(scatterer)
    A=radar.Ai(scatterer)
    return z,zdr,kdp,A

def parsivel_ingest(filename):
    mydata=np.genfromtxt(filename)
    parsivel_dsds=mydata[:,4::]
    timing_data=mydata[:,:4].astype(int)
    datetime_array=np.array([datetime(timing_data[i,0], 
                                  1,1,timing_data[i,2], 
                                  timing_data[i,3]) + \
                                  timedelta(int(timing_data[i,1]-1)) \
                                  for i in range(timing_data.shape[0])])
    drop_diams=np.array([0.064, 0.193, 0.321, 0.45, 0.579, 0.708, 0.836, 
                0.965, 1.094, 1.223, 1.416, 1.674, 1.931, 2.189, 
                2.446, 2.832, 3.347, 3.862, 4.378, 4.892, 5.665, 
                6.695, 7.725, 8.755, 9.785, 11.33, 13.39, 15.45, 
                17.51, 19.57, 22.145, 25.235])
    return datetime_array, drop_diams, parsivel_dsds

all_rd_files = get_file_tree(odir_s, 'csapr_distro*')
all_rd_files.sort()
form = 'csapr_distro_%Y%m%d%H%M%S.txt'
radar_dis_date_list = [datetime.strptime(this_date.split('/')[-1], form) for this_date in all_rd_files]
all_lines = [read_dis(tfile) for tfile in all_rd_files]

ad_form = '%Y%m%d%H%M%S'
actual_dates = np.array([datetime.strptime(this_line.split()[0], ad_form) for this_line in all_lines])
distrometric_history = {}
for i in [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34, 36,38, 40, 42,44]:
    print(i, all_lines[10].split()[i-1])
    this_key = all_lines[10].split()[i-1]
    these_values = np.ma.masked_equal(                                      np.array([masked_conv(this_line.split()[i])                                      for this_line in all_lines]), -9999.0)
    distrometric_history.update({this_key : these_values})

distrometric_history.update({'radar_start':radar_dis_date_list,
                            'overpass_time':actual_dates})
print(len(actual_dates))

all_dis_files = get_file_tree(d_dir, 'sgpvdisC1*')
all_dis_files.sort()

distrodata = netCDF4.Dataset(all_dis_files[50], 'r')
first_distro_units = distrodata.variables['time'].units
d_time = netCDF4.num2date(distrodata.variables['time'][:],
                    first_distro_units)
rwc = distrodata.variables['liquid_water_content'][:]
print(distrodata.variables['liquid_water_content'].units)
print(distrodata.variables['num_density'].units)
distrodata.close()
date_time_list = d_time
rwc_list = rwc
    
for dfile in all_dis_files[51:100]:
    distrodata = netCDF4.Dataset(dfile, 'r')
    d_time = netCDF4.num2date(distrodata.variables['time'][:],
                        distrodata.variables['time'].units)
    rwc = distrodata.variables['liquid_water_content'][:]
    distrodata.close()
    #print(d_time.shape)
    date_time_list = np.concatenate((date_time_list, d_time))
    rwc_list = np.concatenate((rwc_list, rwc))
    

all_tmat_files = get_file_tree(d_dir, '*procc*')
all_tmat_files.sort()
#print(all_tmat_files)

print(all_tmat_files[0])
print(all_dis_files[50])

with open(all_tmat_files[0], 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    myp = u.load()

tmat_kdp = np.array([entry[2] for entry in myp])
tmat_z = np.array([entry[0] for entry in myp])
tmat_zdr = np.array([entry[1] for entry in myp])
tmat_a = np.array([entry[3] for entry in myp])

j = 1
for tfile in all_tmat_files[1::]:
    print(tfile)
    print(all_dis_files[50 + j])
    j = j +1
    with open(tfile, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        myp = u.load()
    
    tmat_kdp = np.concatenate((tmat_kdp, np.array([entry[2] for entry in myp])))
    tmat_z = np.concatenate((tmat_z, np.array([entry[0] for entry in myp])))
    tmat_zdr = np.concatenate((tmat_zdr, np.array([entry[1] for entry in myp])))
    tmat_a = np.concatenate((tmat_a, np.array([entry[3] for entry in myp])))
    

all_par_scatter_files = get_file_tree(p_dir, 'parsivel_apu02*.txtproccessed.pc')
all_par_files = [this_one[0:-13] for this_one in all_par_scatter_files]
all_par_scatter_files.sort()
all_par_files.sort()
print(all_par_files)
with open(all_par_scatter_files[0], 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    myp = u.load()

apu2_tmat_kdp = np.array([entry[2] for entry in myp])
apu2_tmat_z = np.array([entry[0] for entry in myp])
apu2_tmat_zdr = np.array([entry[1] for entry in myp])
apu2_tmat_a = np.array([entry[3] for entry in myp])
apu2_p2_dates, diameters, densities=parsivel_ingest(all_par_files[0])


for i in range(len(all_par_files)-1):
    tfile = all_par_scatter_files[i+1]
    with open(tfile, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        myp = u.load()
    
    these_dates, diameters, densities=parsivel_ingest(all_par_files[i+1])
    apu2_p2_dates = np.concatenate((apu2_p2_dates, these_dates))
    apu2_tmat_kdp = np.concatenate((apu2_tmat_kdp, np.array([entry[2] for entry in myp])))
    apu2_tmat_z = np.concatenate((apu2_tmat_z, np.array([entry[0] for entry in myp])))
    apu2_tmat_zdr = np.concatenate((apu2_tmat_zdr, np.array([entry[1] for entry in myp])))
    apu2_tmat_a = np.concatenate((apu2_tmat_a, np.array([entry[3] for entry in myp])))
    print(tfile)


 

all_par_scatter_files = get_file_tree(p_dir, 'parsivel_apu01*.txtproccessed.pc')
all_par_files = [this_one[0:-13] for this_one in all_par_scatter_files]
all_par_scatter_files.sort()
all_par_files.sort()
print(all_par_files)
with open(all_par_scatter_files[0], 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    myp = u.load()

apu1_tmat_kdp = np.array([entry[2] for entry in myp])
apu1_tmat_z = np.array([entry[0] for entry in myp])
apu1_tmat_zdr = np.array([entry[1] for entry in myp])
apu1_tmat_a = np.array([entry[3] for entry in myp])
apu1_p2_dates, diameters, densities=parsivel_ingest(all_par_files[0])


for i in range(len(all_par_files)-1):
    tfile = all_par_scatter_files[i+1]
    with open(tfile, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        myp = u.load()
    
    these_dates, diameters, densities=parsivel_ingest(all_par_files[i+1])
    apu1_p2_dates = np.concatenate((apu1_p2_dates, these_dates))
    apu1_tmat_kdp = np.concatenate((apu1_tmat_kdp, np.array([entry[2] for entry in myp])))
    apu1_tmat_z = np.concatenate((apu1_tmat_z, np.array([entry[0] for entry in myp])))
    apu1_tmat_zdr = np.concatenate((apu1_tmat_zdr, np.array([entry[1] for entry in myp])))
    apu1_tmat_a = np.concatenate((apu1_tmat_a, np.array([entry[3] for entry in myp])))
    print(tfile)


 



n_mats = len(tmat_kdp)
dist_time_since_first_dist = netCDF4.date2num(date_time_list, 
                                              first_distro_units)
radar_time_since_first_dist = netCDF4.date2num(distrometric_history['radar_start'],
                                              first_distro_units)

KDP_interpolator = interpolate.interp1d(dist_time_since_first_dist, tmat_kdp,
                                        bounds_error=False, fill_value=-9999.0,
                                       kind = 'linear')

z_interpolator = interpolate.interp1d(dist_time_since_first_dist, tmat_z,
                                        bounds_error=False, fill_value=-9999.0,
                                       kind = 'linear')

rwc_interpolator = interpolate.interp1d(dist_time_since_first_dist, rwc_list,
                                        bounds_error=False, fill_value=-9999.0,
                                       kind = 'linear')

T_matrix_kdp_on_radar_points = KDP_interpolator(radar_time_since_first_dist)
T_matrix_z_on_radar_points = z_interpolator(radar_time_since_first_dist)
rwc_on_radar_points = rwc_interpolator(radar_time_since_first_dist)

apu2_time_since_first_dist = netCDF4.date2num(apu2_p2_dates, 
                                              first_distro_units)

apu2_KDP_interpolator = interpolate.interp1d(apu2_time_since_first_dist, apu2_tmat_kdp,
                                        bounds_error=False, fill_value=-9999.0,
                                       kind = 'linear')

apu2_z_interpolator = interpolate.interp1d(apu2_time_since_first_dist, apu2_tmat_z,
                                        bounds_error=False, fill_value=-9999.0,
                                       kind = 'linear')

apu2_T_matrix_kdp_on_radar_points = apu2_KDP_interpolator(radar_time_since_first_dist)
apu2_T_matrix_z_on_radar_points = apu2_z_interpolator(radar_time_since_first_dist)

apu1_time_since_first_dist = netCDF4.date2num(apu1_p2_dates, 
                                              first_distro_units)

apu1_KDP_interpolator = interpolate.interp1d(apu1_time_since_first_dist, apu1_tmat_kdp,
                                        bounds_error=False, fill_value=-9999.0,
                                       kind = 'linear')

apu1_z_interpolator = interpolate.interp1d(apu1_time_since_first_dist, apu1_tmat_z,
                                        bounds_error=False, fill_value=-9999.0,
                                       kind = 'linear')

apu1_T_matrix_kdp_on_radar_points = apu1_KDP_interpolator(radar_time_since_first_dist)
apu1_T_matrix_z_on_radar_points = apu1_z_interpolator(radar_time_since_first_dist)

def find_closest(to_find, find_in):
    costfn = np.abs(find_in - to_find)
    locn = np.where(costfn == costfn.min())[0][0]
    return locn

def block_jumps(jumpy_array, smooth_array, jump_size ):
    delta_time =  np.diff(jumpy_array)
    jumps = np.where(delta_time > jump_size)
    is_a_block = np.ones(len(smooth_array))
    for jump in jumps[0]:
        time_before = jumpy_array[jump]
        location_in_time_before = find_closest(time_before, smooth_array)
        time_after = jumpy_array[jump+1]
        location_in_time_after = find_closest(time_after, smooth_array)
        is_a_block[location_in_time_before:location_in_time_after] = 0.0

    first_time = jumpy_array.min()
    ft_locn = find_closest(first_time, smooth_array)
    last_time = jumpy_array.max()
    lt_locn = find_closest(last_time, smooth_array)
    is_a_block[0:ft_locn] = 0.0
    is_a_block[lt_locn::] = 0.0
    return is_a_block

#create bounds for APUs.. since we have huge regions of missing data
#need to look for big jumps in time

def find_closest(to_find, find_in):
    costfn = np.abs(find_in - to_find)
    locn = np.where(costfn == costfn.min())[0][0]
    return locn

apu1_is_a_block = block_jumps(apu1_time_since_first_dist, radar_time_since_first_dist, 60.0*60.0)
apu2_is_a_block = block_jumps(apu2_time_since_first_dist, radar_time_since_first_dist, 60.0*60.0)

rwc_is_measured = rwc_on_radar_points > 1.
is_correlated = distrometric_history['cross_correlation_ratio'] > 0.85

apu1_active = apu1_is_a_block > 0.1
apu2_active = apu2_is_a_block > 0.1

apu1_valid_kdp = apu1_T_matrix_kdp_on_radar_points > 0.01
apu2_valid_kdp = apu2_T_matrix_kdp_on_radar_points > 0.01

apu1_decent = np.logical_and(apu1_active, apu1_valid_kdp)
apu2_decent = np.logical_and(apu2_active, apu2_valid_kdp)

apu1_decent_and_raining = np.logical_and(apu1_decent, rwc_is_measured)
apu2_decent_and_raining = np.logical_and(apu2_decent, rwc_is_measured)

dvd_kdp_at_least_0p5 = T_matrix_kdp_on_radar_points > 0.1
apu1_kdp_at_least_0p5 = apu1_T_matrix_kdp_on_radar_points > 0.1
apu2_kdp_at_least_0p5 = apu2_T_matrix_kdp_on_radar_points > 0.1


is_good = np.where(np.logical_and(rwc_is_measured, is_correlated))
is_good_apu1 = np.where(apu1_decent)
is_good_apu2 = np.where(apu2_decent)
is_really_good_apu1 = np.where(apu1_decent_and_raining)
is_really_good_apu2 = np.where(apu2_decent_and_raining)

dvd_high = np.logical_and(dvd_kdp_at_least_0p5, rwc_is_measured)
apu1_high = np.logical_and(apu1_kdp_at_least_0p5, apu1_decent_and_raining)
apu2_high = np.logical_and(apu2_kdp_at_least_0p5, apu2_decent_and_raining)


dvd_sig = np.where(dvd_high)
apu1_sig = np.where(apu1_high)
apu2_sig = np.where(apu2_high)
print(len(is_good[0]))
print(len(is_good_apu1[0]))
print(len(is_good_apu2[0]))
print(len(dvd_sig[0]))
print(len(apu1_sig[0]))
print(len(apu2_sig[0]))


alll_t = np.concatenate((T_matrix_kdp_on_radar_points[is_good], 
                       apu1_T_matrix_kdp_on_radar_points[is_really_good_apu1], 
                       apu2_T_matrix_kdp_on_radar_points[is_really_good_apu2]))
alll_r = np.concatenate((distrometric_history['corrected_specific_diff_phase'][is_good], 
                       distrometric_history['corrected_specific_diff_phase'][is_really_good_apu1], 
                       distrometric_history['corrected_specific_diff_phase'][is_really_good_apu2]))

high_alll_t = np.concatenate((T_matrix_kdp_on_radar_points[dvd_sig], 
                       apu1_T_matrix_kdp_on_radar_points[apu1_sig], 
                       apu2_T_matrix_kdp_on_radar_points[apu2_sig]))
high_alll_r = np.concatenate((distrometric_history['corrected_specific_diff_phase'][dvd_sig], 
                       distrometric_history['corrected_specific_diff_phase'][apu1_sig], 
                       distrometric_history['corrected_specific_diff_phase'][apu2_sig]))

dvd_slope, dvd_intercept, dvd_r_value, dvd_p_value, dvd_std_err = stats.linregress(2.0*T_matrix_kdp_on_radar_points[is_good],
                                                                                   distrometric_history['corrected_specific_diff_phase'][is_good])

print(dvd_slope, dvd_intercept)

larger_numbers = np.logical_and(2.0*T_matrix_kdp_on_radar_points[is_good] > 0.1,
                                distrometric_history['corrected_specific_diff_phase'][is_good] > 0.1)

tupleme = stats.linregress(2.0*T_matrix_kdp_on_radar_points[is_good][larger_numbers],
                           distrometric_history['corrected_specific_diff_phase'][is_good][larger_numbers])
dvd_slope_plus, dvd_intercept_plus, dvd_r_value_plus, dvd_p_value_plus, dvd_std_err_plus = tupleme

print(dvd_slope_plus, dvd_intercept_plus)

print(T_matrix_kdp_on_radar_points[dvd_kdp_at_least_0p5])
print(distrometric_history['corrected_specific_diff_phase'][dvd_kdp_at_least_0p5])

all_slope, all_intercept, all_r_value, all_p_value, all_std_err = stats.linregress(alll_t, alll_r)
print(all_slope, all_intercept)
#alls_slope, alls_intercept, alls_r_value, alls_p_value, alls_std_err = stats.linregress(t_radar_at_least_0p5, 
#                                                                                        r_radar_at_least_0p5)
#print(alls_slope, alls_intercept)
#r_radar_at_least_0p5

print(high_alll_t)
print(high_alll_r)

high_slope, high_intercept, high_r_value, high_p_value, high_std_err = stats.linregress(high_alll_t, high_alll_r)
print(high_slope, high_intercept)

model = sm.OLS(alll_t, 
                alll_r)
results = model.fit()
print(results.params)
results.rsquared

print(len(apu1_T_matrix_kdp_on_radar_points))
print(len(distrometric_history['corrected_specific_diff_phase']))

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

rc('font', **font)

plt.figure(figsize = [10,7])
kdp_bins = np.linspace(0,4,17)
H, xe, ye = np.histogram2d(T_matrix_kdp_on_radar_points[is_good],
         distrometric_history['corrected_specific_diff_phase'][is_good], bins=(20, 20), range=([0,1], [0,1]))

plt.pcolormesh(xe, ye, H.transpose(), norm=LogNorm(vmin=1), vmax=100)
plt.plot(T_matrix_kdp_on_radar_points[is_good],
         distrometric_history['corrected_specific_diff_phase'][is_good], 
         'yo', alpha = 1)

#plt.plot(apu1_T_matrix_kdp_on_radar_points[is_really_good_apu1]*2.0,
#         distrometric_history['corrected_specific_diff_phase'][is_really_good_apu1], 
#         'ro', alpha = 0.5)

#plt.plot(apu2_T_matrix_kdp_on_radar_points[is_really_good_apu2]*2.0,
#         distrometric_history['corrected_specific_diff_phase'][is_really_good_apu2], 
#         'bo', alpha = 0.5)

plt.ylim([0,1])
plt.xlim([0,1])
plt.xlabel(r'T-Matrix 2DVD K$_{DP}$')
plt.ylabel(r'LP Method Radar K$_{DP}$')

plt.colorbar(label='Number of points')

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

rc('font', **font)

plt.figure(figsize = [10,7])
kdp_bins = np.linspace(0,4,17)
H, xe, ye = np.histogram2d(T_matrix_kdp_on_radar_points[is_good],
         distrometric_history['corrected_specific_diff_phase'][is_good], bins=(21, 21), range=([0,2], [0,2]))

plt.pcolormesh(xe, ye, H.transpose(), norm=LogNorm(vmin=1), vmax=100)
plt.plot(T_matrix_kdp_on_radar_points[is_good],
         distrometric_history['corrected_specific_diff_phase'][is_good], 
         'yo', alpha = 1)

#plt.plot(apu1_T_matrix_kdp_on_radar_points[is_really_good_apu1]*2.0,
#         distrometric_history['corrected_specific_diff_phase'][is_really_good_apu1], 
#         'ro', alpha = 0.5)

#plt.plot(apu2_T_matrix_kdp_on_radar_points[is_really_good_apu2]*2.0,
#         distrometric_history['corrected_specific_diff_phase'][is_really_good_apu2], 
#         'bo', alpha = 0.5)

plt.ylim([0,2])
plt.xlim([0,2])
plt.xlabel(r'T-Matrix 2DVD K$_{DP}$')
plt.ylabel(r'LP Method Radar K$_{DP}$')

plt.colorbar(label='Number of points')

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

rc('font', **font)

plt.figure(figsize = [10,8])
kdp_bins = np.linspace(0,4,17)
H, xe, ye = np.histogram2d(distrometric_history['corrected_specific_diff_phase'][is_good],
         distrometric_history['bringi_specific_diff_phase'][is_good], bins=(20, 20), range=([0,2], [0,2]))

plt.pcolormesh(xe, ye, H.transpose(), norm=LogNorm(vmin=1), vmax=100)
plt.plot(distrometric_history['corrected_specific_diff_phase'][is_good],
         distrometric_history['bringi_specific_diff_phase'][is_good], 
         'yo', alpha = 1)

#plt.plot(apu1_T_matrix_kdp_on_radar_points[is_really_good_apu1]*2.0,
#         distrometric_history['corrected_specific_diff_phase'][is_really_good_apu1], 
#         'ro', alpha = 0.5)

#plt.plot(apu2_T_matrix_kdp_on_radar_points[is_really_good_apu2]*2.0,
#         distrometric_history['corrected_specific_diff_phase'][is_really_good_apu2], 
#         'bo', alpha = 0.5)

plt.ylim([0,2])
plt.xlim([0,2])
plt.xlabel(r'LP Method Radar K$_{DP}$')
plt.ylabel(r'Bringi Method Radar K$_{DP}$')

plt.colorbar(label='Number of points')

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

rc('font', **font)

plt.figure(figsize = [10,8])
kdp_bins = np.linspace(0,4,17)
H, xe, ye = np.histogram2d(T_matrix_kdp_on_radar_points[is_good],
         distrometric_history['bringi_specific_diff_phase'][is_good], bins=(20, 20), range=([0,2], [0,2]))

plt.pcolormesh(xe, ye, H.transpose(), norm=LogNorm(vmin=1), vmax=100)
plt.plot(T_matrix_kdp_on_radar_points[is_good],
         distrometric_history['bringi_specific_diff_phase'][is_good], 
         'yo', alpha = 1)

#plt.plot(apu1_T_matrix_kdp_on_radar_points[is_really_good_apu1]*2.0,
#         distrometric_history['corrected_specific_diff_phase'][is_really_good_apu1], 
#         'ro', alpha = 0.5)

#plt.plot(apu2_T_matrix_kdp_on_radar_points[is_really_good_apu2]*2.0,
#         distrometric_history['corrected_specific_diff_phase'][is_really_good_apu2], 
#         'bo', alpha = 0.5)

plt.ylim([0,2])
plt.xlim([0,2])
plt.xlabel(r'T-Matrix 2DVD K$_{DP}$')
plt.ylabel(r'Bringi Method Radar K$_{DP}$')

plt.colorbar(label='Number of points')

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

rc('font', **font)

plt.figure(figsize = [10,8])
kdp_bins = np.linspace(0,4,17)
H, xe, ye = np.histogram2d(T_matrix_kdp_on_radar_points[is_good],
         distrometric_history['maesaka_differential_phase'][is_good], bins=(20, 20), range=([0,2], [0,2]))

plt.pcolormesh(xe, ye, H.transpose(), norm=LogNorm(vmin=1), vmax=100)
plt.plot(T_matrix_kdp_on_radar_points[is_good],
         distrometric_history['maesaka_differential_phase'][is_good], 
         'yo', alpha = 1)

#plt.plot(apu1_T_matrix_kdp_on_radar_points[is_really_good_apu1]*2.0,
#         distrometric_history['corrected_specific_diff_phase'][is_really_good_apu1], 
#         'ro', alpha = 0.5)

#plt.plot(apu2_T_matrix_kdp_on_radar_points[is_really_good_apu2]*2.0,
#         distrometric_history['corrected_specific_diff_phase'][is_really_good_apu2], 
#         'bo', alpha = 0.5)

plt.ylim([0,2])
plt.xlim([0,2])
plt.xlabel(r'T-Matrix 2DVD K$_{DP}$')
plt.ylabel(r'Bringi Method Radar K$_{DP}$')

plt.colorbar(label='Number of points')

time1=mdates.datestr2num('20110520 1000')
time2=mdates.datestr2num('20110520 1510')

hours = mdates.HourLocator()
days = mdates.HourLocator(interval = 2)
tFmt = mdates.DateFormatter('%H')
f = plt.figure(figsize=[12,5])
plt.subplot(2,1,1)
plt.plot(mdates.date2num(distrometric_history['radar_start']),
         distrometric_history['corrected_specific_diff_phase'])
ax = plt.gca()
plt.ylabel('Radar KDP')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
plt.ylim([0,6])
plt.subplot(2,1,2)
plt.plot(mdates.date2num(distrometric_history['radar_start']),
         apu1_T_matrix_kdp_on_radar_points)
ax = plt.gca()
plt.ylabel('Twice TMatrix KDP')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
plt.ylim([0,6])
plt.xlabel('Time on May 20th 2011 (UTC)')


time1=mdates.datestr2num('20110420 1000')
time2=mdates.datestr2num('20110602 1510')

hours = mdates.HourLocator()
days = mdates.DayLocator(interval = 2)
tFmt = mdates.DateFormatter('%d')
f = plt.figure(figsize=[12,5])
plt.subplot(2,1,1)
plt.plot(mdates.date2num(distrometric_history['radar_start']),
         distrometric_history['corrected_specific_diff_phase'])
ax = plt.gca()
plt.ylabel('Radar KDP')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
plt.ylim([0,1])
plt.subplot(2,1,2)
plt.plot(mdates.date2num(distrometric_history['radar_start']),
         apu1_T_matrix_kdp_on_radar_points)
ax = plt.gca()
plt.ylabel('Twice TMatrix KDP')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
plt.ylim([0,1])
plt.xlabel('Time on May 20th 2011 (UTC)')


plt.plot(distrometric_history['corrected_specific_diff_phase'][is_good], 
         T_matrix_kdp_on_radar_points[is_good]*2.0, 'ro')
plt.ylim([0,4])
plt.xlim([0,4])

plt.plot(distrometric_history['corrected_specific_diff_phase'][is_good_apu1], 
         apu1_T_matrix_kdp_on_radar_points[is_good_apu1]*2.0, 'ro')
plt.ylim([0,4])
plt.xlim([0,4])

plt.plot(distrometric_history['corrected_specific_diff_phase'][is_good], 
         apu1_T_matrix_kdp_on_radar_points[is_good]*2, 'ro')
plt.ylim([0,4])
plt.xlim([0,4])

kdp_bins = np.linspace(0,4,17)
H, xe, ye = np.histogram2d(distrometric_history['corrected_specific_diff_phase'][is_good], 
         T_matrix_kdp_on_radar_points[is_good]*2.0, bins=(16, 16), range=([0,4], [0,4]))
plt.pcolormesh(xe, ye, H, norm=LogNorm(vmin=0.1, vmax=H.max()), cmap='PuBu_r')
plt.colorbar()

kdp_bins = np.linspace(0,4,17)
H, xe, ye = np.histogram2d(distrometric_history['corrected_specific_diff_phase'][is_good], 
         T_matrix_kdp_on_radar_points[is_good]*2.0, bins=(16, 16), range=([0,1], [0,1]))
plt.pcolormesh(xe, ye, H, norm=LogNorm(vmin=0.1, vmax=H.max()), cmap='PuBu_r')
plt.colorbar()

kdp_bins = np.linspace(0,4,17)
H, xe, ye = np.histogram2d(alll_r, 
         alll_t*2.0, bins=(16, 16), range=([0,4], [0,4]))
plt.pcolormesh(xe, ye, H, norm=LogNorm(vmin=0.1, vmax=H.max()), cmap='PuBu_r')
plt.colorbar()

kdp_bins = np.linspace(0,4,17)
H, xe, ye = np.histogram2d(alll_r, 
         alll_t*2.0, bins=(10, 10), range=([0,1], [0,1]))
plt.pcolormesh(xe, ye, H, norm=LogNorm(vmin=0.1, vmax=H.max()), cmap='PuBu_r')
plt.colorbar()

plt.plot(alll_r, 
         alll_t*2.0, 'ro')
plt.ylim([0,4])
plt.xlim([0,4])

plt.plot(alll_r, 
         alll_t*2.0, 'ro')
plt.ylim([0,1])
plt.xlim([0,1])

kdp_bins = np.linspace(0,4,17)
H, xe, ye = np.histogram2d(alll_r, 
         alll_t*2.0, bins=(10, 10), range=([0,4], [0,4]))
plt.pcolormesh(xe, ye, H, norm=LogNorm(vmin=0.01), vmax=100)
plt.plot(distrometric_history['corrected_specific_diff_phase'][is_good], 
         T_matrix_kdp_on_radar_points[is_good]*2.0, 'k.', alpha = 0.5)

plt.plot(distrometric_history['corrected_specific_diff_phase'][is_really_good_apu1], 
         apu1_T_matrix_kdp_on_radar_points[is_really_good_apu1]*2.0, 'r.', alpha = 0.5)

plt.plot(distrometric_history['corrected_specific_diff_phase'][is_really_good_apu2], 
         apu2_T_matrix_kdp_on_radar_points[is_really_good_apu2]*2.0, 'b.', alpha = 0.5)

plt.ylim([0,4])
plt.xlim([0,4])
plt.colorbar()

kdp_bins = np.linspace(0,4,17)
H, xe, ye = np.histogram2d(alll_r, 
         alll_t*2.0, bins=(10, 10), range=([0,4], [0,4]))
plt.pcolormesh(xe, ye, H, norm=LogNorm(vmin=0.01), vmax=100)
plt.plot(distrometric_history['bringi_specific_diff_phase'][is_good], 
         T_matrix_kdp_on_radar_points[is_good]*2.0, 'k.', alpha = 0.5)

plt.plot(distrometric_history['bringi_specific_diff_phase'][is_really_good_apu1], 
         apu1_T_matrix_kdp_on_radar_points[is_really_good_apu1]*2.0, 'r.', alpha = 0.5)

plt.plot(distrometric_history['bringi_specific_diff_phase'][is_really_good_apu2], 
         apu2_T_matrix_kdp_on_radar_points[is_really_good_apu2]*2.0, 'b.', alpha = 0.5)

plt.ylim([0,4])
plt.xlim([0,4])
plt.colorbar()

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

rc('font', **font)

plt.figure(figsize = [14,10])
kdp_bins = np.linspace(0,4,17)
H, xe, ye = np.histogram2d(alll_r, 
         alll_t*2.0, bins=(20, 20), range=([0,1], [0,1]))
plt.pcolormesh(xe, ye, H, norm=LogNorm(vmin=1), vmax=100)
plt.plot(distrometric_history['bringi_specific_diff_phase'][is_good], 
         T_matrix_kdp_on_radar_points[is_good]*2.0, 'ko', alpha = 0.5)

plt.plot(distrometric_history['bringi_specific_diff_phase'][is_really_good_apu1], 
         apu1_T_matrix_kdp_on_radar_points[is_really_good_apu1]*2.0, 'ro', alpha = 0.5)

plt.plot(distrometric_history['bringi_specific_diff_phase'][is_really_good_apu2], 
         apu2_T_matrix_kdp_on_radar_points[is_really_good_apu2]*2.0, 'bo', alpha = 0.5)

plt.ylim([0,1])
plt.xlim([0,1])
plt.colorbar()

time1=mdates.datestr2num('20110424 0700')
time2=mdates.datestr2num('20110610 1210')

hours = mdates.HourLocator()
days = mdates.DayLocator(interval = 4)
tFmt = mdates.DateFormatter('%m/%d')
f = plt.figure(figsize=[12,5])
plt.plot(mdates.date2num(distrometric_history['radar_start']),
         distrometric_history['reflectivity'])
ax = plt.gca()
#plt.ylabel('Drop volume Density $\mathrm{(m^{-3})}$')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
plt.xlabel('Time on May 20th 2011 (UTC)')


time1=mdates.datestr2num('20110424 0700')
time2=mdates.datestr2num('20110603 1210')

hours = mdates.HourLocator()
days = mdates.DayLocator(interval = 4)
tFmt = mdates.DateFormatter('%m/%d')
f = plt.figure(figsize=[15,5])
plt.plot(mdates.date2num(distrometric_history['radar_start']),
         distrometric_history['velocity_texture'])
ax = plt.gca()
#plt.ylabel('Drop volume Density $\mathrm{(m^{-3})}$')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
plt.xlabel('Time on May 20th 2011 (UTC)')


time1=mdates.datestr2num('20110424 0000')
time2=mdates.datestr2num('20110602 1210')

hours = mdates.HourLocator()
days = mdates.DayLocator(interval = 4)
tFmt = mdates.DateFormatter('%m/%d')
f = plt.figure(figsize=[12,5])
plt.plot(mdates.date2num(distrometric_history['radar_start']),
         distrometric_history['cross_correlation_ratio'])
ax = plt.gca()
#plt.ylabel('Drop volume Density $\mathrm{(m^{-3})}$')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
plt.xlabel('Time on May 20th 2011 (UTC)')


time1=mdates.datestr2num('20110424 0000')
time2=mdates.datestr2num('20110602 1210')

hours = mdates.HourLocator()
days = mdates.DayLocator(interval = 4)
tFmt = mdates.DateFormatter('%m/%d')
f = plt.figure(figsize=[12,5])
plt.plot(mdates.date2num(distrometric_history['radar_start']),
         distrometric_history['gate_id'])
ax = plt.gca()
#plt.ylabel('Drop volume Density $\mathrm{(m^{-3})}$')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
plt.xlabel('Time on May 20th 2011 (UTC)')

#0: no_scatter 1: melting 2: multi_trip 3: rain 4: snow 

time1=mdates.datestr2num('20110424 0700')
time2=mdates.datestr2num('20110610 1210')

hours = mdates.HourLocator()
days = mdates.DayLocator(interval = 4)
tFmt = mdates.DateFormatter('%m/%d')
f = plt.figure(figsize=[12,5])
plt.plot(mdates.date2num(distrometric_history['radar_start']),
         distrometric_history['corrected_specific_diff_phase'])
ax = plt.gca()
#plt.ylabel('Drop volume Density $\mathrm{(m^{-3})}$')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
plt.xlabel('Time on May 20th 2011 (UTC)')


time1=mdates.datestr2num('20110519 0700')
time2=mdates.datestr2num('20110521 1210')

hours = mdates.HourLocator()
days = mdates.DayLocator(interval = 4)
tFmt = mdates.DateFormatter('%m/%d')
f = plt.figure(figsize=[12,5])
plt.plot(mdates.date2num(distrometric_history['radar_start']),
         distrometric_history['reflectivity'])
ax = plt.gca()
#plt.ylabel('Drop volume Density $\mathrm{(m^{-3})}$')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
plt.xlabel('Time on May 20th 2011 (UTC)')


time1=mdates.datestr2num('20110425 0000')
time2=mdates.datestr2num('20110427 1210')

hours = mdates.HourLocator()
days = mdates.DayLocator(interval = 4)
tFmt = mdates.DateFormatter('%m/%d')
f = plt.figure(figsize=[12,5])
plt.plot(mdates.date2num(distrometric_history['radar_start']),
         distrometric_history['corrected_specific_diff_phase'])
ax = plt.gca()
#plt.ylabel('Drop volume Density $\mathrm{(m^{-3})}$')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
plt.xlabel('Time on May 20th 2011 (UTC)')


time1=mdates.datestr2num('20110519 0700')
time2=mdates.datestr2num('20110521 1210')

hours = mdates.HourLocator()
days = mdates.DayLocator(interval = 4)
tFmt = mdates.DateFormatter('%m/%d')
f = plt.figure(figsize=[12,5])
plt.plot(mdates.date2num(distrometric_history['radar_start']),
         distrometric_history['corrected_specific_diff_phase'])
ax = plt.gca()
#plt.ylabel('Drop volume Density $\mathrm{(m^{-3})}$')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
plt.xlabel('Time on May 20th 2011 (UTC)')








time1=mdates.datestr2num('20110424 0700')
time2=mdates.datestr2num('20110610 1210')

hours = mdates.HourLocator()
days = mdates.DayLocator(interval = 4)
tFmt = mdates.DateFormatter('%m/%d')
f = plt.figure(figsize=[12,5])
plt.semilogy(mdates.date2num(date_time_list),
         rwc_list+ 0.01)
ax = plt.gca()
#plt.ylabel('Drop volume Density $\mathrm{(m^{-3})}$')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
plt.xlabel('Time on May 20th 2011 (UTC)')





print(all_dis_files[99])
print(len(date_time_list))
print(len(tmat_kdp))

n_mats = len(tmat_kdp)
time1=mdates.datestr2num('20110424 0700')
time2=mdates.datestr2num('20110610 1210')
hours = mdates.HourLocator()
days = mdates.DayLocator(interval = 2)
tFmt = mdates.DateFormatter('%d')
f = plt.figure(figsize=[18,5])
plt.subplot(2,1,1)

plt.plot(mdates.date2num(date_time_list[0:n_mats]),
         tmat_kdp)
ax = plt.gca()
#plt.ylabel('Drop volume Density $\mathrm{(m^{-3})}$')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
#plt.xlabel('Time on May 20th 2011 (UTC)')
plt.ylim([0,5])
plt.ylabel(r'T-Matrix calculated K$_{dp}$')
plt.subplot(2,1,2)

plt.semilogy(mdates.date2num(date_time_list),
         rwc_list+0.1)
ax = plt.gca()
#plt.ylabel('Drop volume Density $\mathrm{(m^{-3})}$')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
plt.xlabel('Time on May 20th 2011 (UTC)')
plt.ylabel(r'Rain water content (mm$^3$/m$^3$)')


n_mats = len(tmat_kdp)
time1=mdates.datestr2num('20110517 0700')
time2=mdates.datestr2num('20110519 1210')
hours = mdates.HourLocator()
days = mdates.DayLocator(interval = 1)
tFmt = mdates.DateFormatter('%m/%d')
f = plt.figure(figsize=[18,5])
plt.subplot(2,1,1)

plt.plot(mdates.date2num(date_time_list),
         tmat_kdp)
ax = plt.gca()
#plt.ylabel('Drop volume Density $\mathrm{(m^{-3})}$')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
#plt.xlabel('Time on May 20th 2011 (UTC)')

plt.subplot(2,1,2)

plt.semilogy(mdates.date2num(date_time_list),
         rwc_list+0.1)
ax = plt.gca()
#plt.ylabel('Drop volume Density $\mathrm{(m^{-3})}$')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
plt.xlabel('Time on May 20th 2011 (UTC)')


n_mats = len(tmat_kdp)
time1=mdates.datestr2num('20110424 0700')
time2=mdates.datestr2num('20110610 1210')

hours = mdates.HourLocator()
days = mdates.DayLocator(interval = 4)
tFmt = mdates.DateFormatter('%m/%d')
f = plt.figure(figsize=[12,5])
plt.plot(mdates.date2num(date_time_list),
         10.0*np.log10(tmat_z +0.0001))
ax = plt.gca()
#plt.ylabel('Drop volume Density $\mathrm{(m^{-3})}$')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
plt.xlabel('Time on May 20th 2011 (UTC)')

time1=mdates.datestr2num('20110520 0700')
time2=mdates.datestr2num('20110520 1210')

hours = mdates.HourLocator()
days = mdates.DayLocator(interval = 4)
tFmt = mdates.DateFormatter('%m/%d')
f = plt.figure(figsize=[12,5])
plt.plot(mdates.date2num(apu1_p2_dates),
         10.0*np.log10(apu1_tmat_z +0.0001))
ax = plt.gca()
#plt.ylabel('Drop volume Density $\mathrm{(m^{-3})}$')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
plt.xlabel('Time on May 20th 2011 (UTC)')

time1=mdates.datestr2num('20110520 0700')
time2=mdates.datestr2num('20110520 1210')

hours = mdates.HourLocator()
days = mdates.DayLocator(interval = 4)
tFmt = mdates.DateFormatter('%m/%d')
f = plt.figure(figsize=[12,5])
plt.plot(mdates.date2num(apu1_p2_dates),
         apu1_tmat_kdp)
ax = plt.gca()
#plt.ylabel('Drop volume Density $\mathrm{(m^{-3})}$')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
plt.xlabel('Time on May 20th 2011 (UTC)')

print(first_distro_units)



time1=mdates.datestr2num('20110424 0700')
time2=mdates.datestr2num('20110610 1210')

hours = mdates.HourLocator()
days = mdates.DayLocator(interval = 4)
tFmt = mdates.DateFormatter('%m/%d')
f = plt.figure(figsize=[12,5])
plt.subplot(2,1,1)
plt.plot(mdates.date2num(distrometric_history['radar_start']),
         distrometric_history['bringi_specific_diff_phase'])
ax = plt.gca()
#plt.ylabel('Drop volume Density $\mathrm{(m^{-3})}$')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
plt.ylim([0,6])
plt.subplot(2,1,2)
plt.plot(mdates.date2num(distrometric_history['radar_start']),
         T_matrix_kdp_on_radar_points*2.0)
ax = plt.gca()
#plt.ylabel('Drop volume Density $\mathrm{(m^{-3})}$')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
plt.ylim([0,6])
plt.xlabel('Time on May 20th 2011 (UTC)')


time1=mdates.datestr2num('20110520 1000')
time2=mdates.datestr2num('20110520 1510')

hours = mdates.HourLocator()
days = mdates.HourLocator(interval = 2)
tFmt = mdates.DateFormatter('%H')
f = plt.figure(figsize=[12,5])
plt.subplot(2,1,1)
plt.plot(mdates.date2num(distrometric_history['radar_start']),
         distrometric_history['corrected_specific_diff_phase'])
ax = plt.gca()
plt.ylabel('Radar KDP')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
plt.ylim([0,6])
plt.subplot(2,1,2)
plt.plot(mdates.date2num(distrometric_history['radar_start']),
         T_matrix_kdp_on_radar_points*2)
ax = plt.gca()
plt.ylabel('Twice TMatrix KDP')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
plt.ylim([0,6])
plt.xlabel('Time on May 20th 2011 (UTC)')


time1=mdates.datestr2num('20110520 1000')
time2=mdates.datestr2num('20110520 1510')

hours = mdates.HourLocator()
days = mdates.HourLocator(interval = 2)
tFmt = mdates.DateFormatter('%H')
f = plt.figure(figsize=[12,5])
plt.subplot(2,1,1)
plt.plot(mdates.date2num(distrometric_history['radar_start']),
         distrometric_history['reflectivity'])
ax = plt.gca()
plt.ylabel('Z from Radar')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
plt.ylim([0,50])
plt.subplot(2,1,2)
plt.plot(mdates.date2num(distrometric_history['radar_start']),
         10.0*np.log10(T_matrix_z_on_radar_points+0.0001))
ax = plt.gca()
plt.ylabel('Z from T-Matrix')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
plt.ylim([0,50])
plt.xlabel('Time on May 20th 2011 (UTC)')


n_mats = len(tmat_kdp)
time1=mdates.datestr2num('20110520 0700')
time2=mdates.datestr2num('20110520 1210')
hours = mdates.HourLocator()
days = mdates.HourLocator(interval = 2)
tFmt = mdates.DateFormatter('%H')
f = plt.figure(figsize=[18,5])
plt.subplot(2,1,1)

plt.plot(mdates.date2num(date_time_list[0:n_mats]),
         tmat_kdp)
ax = plt.gca()
#plt.ylabel('Drop volume Density $\mathrm{(m^{-3})}$')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
#plt.xlabel('Time on May 20th 2011 (UTC)')
plt.ylim([0,5])
plt.ylabel(r'T-Matrix calculated K$_{dp}$')
plt.subplot(2,1,2)
plt.plot(mdates.date2num(distrometric_history['radar_start']),
         distrometric_history['corrected_specific_diff_phase'])

ax = plt.gca()
#plt.ylabel('Drop volume Density $\mathrm{(m^{-3})}$')
ax.xaxis_date()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(tFmt)
ax.set_xlim([time1, time2])
plt.xlabel('Time on May 20th 2011 (UTC)')
#plt.ylabel(r'Rain water content (mm$^3$/m$^3$)')






plt.plot(distrometric_history['corrected_specific_diff_phase'][is_good], 
         T_matrix_kdp_on_radar_points[is_good]*2.0, 'ro')
plt.ylim([0,4])
plt.xlim([0,4])

plt.plot(distrometric_history['bringi_specific_diff_phase'][is_good], 
         T_matrix_kdp_on_radar_points[is_good]*2.0, 'ro')
plt.ylim([0,4])
plt.xlim([0,4])

plt.plot(distrometric_history['maesaka_differential_phase'][is_good], 
         T_matrix_kdp_on_radar_points[is_good]*2.0, 'ro')
plt.ylim([0,1])
plt.xlim([0,1])

plt.plot(distrometric_history['maesaka_differential_phase'])

print(2000*6./60)

200/(12*12)







