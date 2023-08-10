from astropy.io import fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.time import Time
get_ipython().magic('matplotlib inline')



cwd = os.getcwd()
work_dir = cwd+'/data/G0096_67/2006d272'
os.chdir(work_dir)

ls

if not os.path.exists('observation_list.txt'):
    os.system('ls *.fits > fits_list.txt')
    file_list = np.atleast_1d(np.loadtxt('fits_list.txt', dtype='string'))
    with open('observation_list.txt', 'w') as f:
        for filename in file_list:
            try:
                hdulist = pyfits.open(filename)
                position = hdulist[2].header['POSITION']
                pointing = str(hdulist[0].header['SPPOINT'])
                description = ' '.join([filename, position, pointing, '\n'])
                f.write(description)
            except IOError:
                print('Empty or corrupt file: %s'%filename)
    os.system('rm fits_list.txt')
    

file_list = np.loadtxt(fname='observation_list.txt', delimiter=' ', 
                       dtype={'names' : ('filename', 'position', 'pointing'),
                             'formats' : ('S100', 'S3', 'b')})
file_list

def load_file(filename):
    """
    Open fits file, load spectra and extract information
    relevant to reductions.
    Returns a dictionary object.
    """
    hdulist = pyfits.open(filename)
    spectrum = {'Vlsr' : hdulist[2].data['Vlsr'],
                'LCP'  : hdulist[2].data['Polstate1'],
                'RCP'  : hdulist[2].data['Polstate4'],
                'units' : hdulist[2].header['TUNIT2'],
                'pointing' : hdulist[0].header['SPPOINT'],
                'position' : hdulist[2].header['POSITION'], 
                'HPBW' : hdulist[1].header['HPBW'],
                'date' : hdulist[0].header['DATE-OBS'],
                'object' : hdulist[0].header['OBJECT'],
                'longitude' : hdulist[0].header['LONGITUD'],
                'latitude' : hdulist[0].header['LATITUDE'],
                'equinox' : hdulist[0].header['EQUINOX'],
                'bw' : hdulist[0].header['SPBW'],
                'nchan' : hdulist[0].header['SPCHAN'],
                'integration' : hdulist[0].header['SPTIME'],
                'fs_offset' : hdulist[0].header['SPFS'],
                'spVlsr' : hdulist[0].header['SPVLSR'],
                'restfreq' : hdulist[0].header['RESTFREQ'],
                'centrefreq' : hdulist[2].header['CENTFREQ'], 
                'Tsys_lcp' : hdulist[3].header['TSYS1'],
                'DTsys_lcp' : hdulist[3].header['TSYSERR1'],
                'Tsys_rcp' : hdulist[3].header['TSYS2'],
                'DTsys_rcp': hdulist[3].header['TSYSERR2'] }
    return spectrum

#get indices of the on-source files
full_int_on = np.where(np.logical_and(file_list['position']=='ON', file_list['pointing']==0))[0]
on_fs1 = load_file(file_list['filename'][full_int_on[0]])
on_fs2 = load_file(file_list['filename'][full_int_on[1]])

def plot_spec(spec):
    """ Produce spectral plots given spectral line data"""
    plt.figure(figsize=[15,3])
    plt.plot(spec['Vlsr'], spec['LCP'], label='LCP', linewidth=3)
    plt.plot(spec['Vlsr'], spec['RCP'], label='RCP', linewidth=3)
    plt.legend()
    plt.title(spec['object'] + ' ' + spec['date'])
    plt.xlabel('V_lsr (km/s)')
    plt.ylabel('Antenna temperature (K)')
    plt.axis('tight')

plot_spec(on_fs1)
plot_spec(on_fs2)

lcp1 = on_fs1['LCP'] - on_fs2['LCP']
lcp2 = on_fs2['LCP'] - on_fs1['LCP']
plt.figure(figsize=[20,10])
ax = plt.subplot(121)
plt.plot(on_fs1['Vlsr'], lcp1, color='r', linewidth=3)
lim=plt.axis('tight')
plt.xlabel('V_lsr (km/s)')
plt.ylabel('Antenna temperature (K)')
plt.title('First observation')
ax = plt.subplot(122)
plt.plot(on_fs2['Vlsr'], lcp2, color='b', linewidth=3)
lim=plt.axis('tight')
plt.xlabel('V_lsr (km/s)')
plt.ylabel('Antenna temperature (K)')
plt.title('Second observation')
plt.figure(figsize=[20,10])
plt.plot(on_fs1['Vlsr'], lcp1, color='r', linewidth=3, alpha=0.5)
plt.plot(on_fs2['Vlsr'], lcp2, color='b', linewidth=3, alpha=0.5)
lim=plt.axis('tight')
plt.xlabel('V_lsr (km/s)')
plt.ylabel('Antenna temperature (K)')
plt.title('Combined observation showing velocity overlap')


#where do the velocities intersect?
spec1_common = np.in1d(on_fs1['Vlsr'], on_fs2['Vlsr'])
spec2_common = np.in1d(on_fs2['Vlsr'], on_fs1['Vlsr'])

spec1_common

spec2_common

spec1_common = np.nonzero(np.in1d(on_fs1['Vlsr'], on_fs2['Vlsr']))
spec2_common = np.nonzero(np.in1d(on_fs2['Vlsr'], on_fs1['Vlsr']))
print spec1_common
print spec2_common

plt.plot(on_fs1['Vlsr'][spec1_common], lcp1[spec1_common], label = 'first spectrum')
plt.plot(on_fs2['Vlsr'][spec2_common], lcp2[spec2_common], label = 'second spectrum')
plt.legend()
plt.xlabel('V_lsr (km/s)')
plt.ylabel('Antenna temperature (K)')

def freq_switch(spec1, spec2):
    """
    Correct for bandpass response using frequency-switching and
    return spectra over common velocity range.
    Inputs: 
    spec1 : first observation of frequency-switched pair
    spec2 : second observation of frequency-switched pair
    Returns:
    new_spec1: bandpass-corrected frequency range
    new_spec2: bandpass-corrected frequency range
    
    """
    #subtract reference spectrum from signal
    lcp1 = spec1['LCP'] - spec2['LCP']
    lcp2 = spec2['LCP'] - spec1['LCP']
    rcp1 = spec1['RCP'] - spec2['RCP']
    rcp2 = spec2['RCP'] - spec1['RCP']
    
    #find the indices of the common velocity channels for each spectrum
    spec1_common = np.nonzero(np.in1d(spec1['Vlsr'], spec2['Vlsr']))
    spec2_common = np.nonzero(np.in1d(spec2['Vlsr'], spec1['Vlsr']))
    
    #make a copy of the input spectra to preserve the header information
    new_spec1 = spec1.copy()
    new_spec2 = spec2.copy()
    
    #insert the new spectra into the copies
    new_spec1['Vlsr'] = spec1['Vlsr'][spec1_common]
    new_spec1['LCP'] = lcp1[spec1_common]
    new_spec1['RCP'] = rcp1[spec1_common]
    
    new_spec2['Vlsr'] = spec2['Vlsr'][spec2_common]
    new_spec2['LCP'] = lcp2[spec2_common]
    new_spec2['RCP'] = rcp2[spec2_common]

    
    return new_spec1, new_spec2

new_spec1, new_spec2 = freq_switch(on_fs1, on_fs2)
plot_spec(new_spec1)
plot_spec(new_spec2)

plt.figure(figsize = [15,10])
ax = plt.subplot(211)
ax.plot(new_spec1['Vlsr'], new_spec1['LCP'], linewidth=3, label = 'Spectrum 1')
plt.axis('tight')
plt.ylim(-5, 10)
ax.axhline(0,color='k')
plt.legend()
#plt.xlabel('Vlsr (km/s)', fontsize=20)
#plt.ylabel('Antenna temperature (K)', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
start, end = ax.get_xlim()
tick_interval = 2
ticks = ax.xaxis.set_ticks(np.arange(start, end, tick_interval))  # makes it easier to get velocity ranges
ax = plt.subplot(212)
ax.plot(new_spec2['Vlsr'], new_spec2['LCP'], linewidth=3, label = 'Spectrum 2')
plt.axis('tight')
plt.ylim(-5, 10)
ax.axhline(0,color='k')
plt.xlabel('Vlsr (km/s)', fontsize=20)
plt.ylabel('                    Antenna temperature (K)', fontsize=20)
plt.legend()
start, end = ax.get_xlim()
tick_interval = 2
ticks = ax.xaxis.set_ticks(np.arange(start, end, tick_interval))  # makes it easier to get velocity ranges

#define a range over which to fit fit the baseline.
#We exclude the channels containing line emission.
line_range = [-2.78, 7.22]
line_chans = np.where(np.logical_and(new_spec1['Vlsr']>=line_range[0], new_spec1['Vlsr']<=line_range[1]))
mask = np.zeros(len(new_spec1['Vlsr']))
mask[line_chans] = 1
vel_mask = np.ma.array(new_spec1['Vlsr'], mask = mask)
amp_mask = np.ma.array(new_spec1['LCP'], mask = mask)

#fit a third-order polynomial to the baseline
p = np.poly1d(np.ma.polyfit(vel_mask, amp_mask,  3))
basefit = p(new_spec1['Vlsr'])

#inspect the fit
plt.figure(figsize = [15,10])
ax = plt.subplot(211)
plt.plot(new_spec1['Vlsr'], new_spec1['LCP'], linewidth=3)
plt.plot(vel_mask, amp_mask, linewidth=3)
plt.plot(new_spec1['Vlsr'], basefit, linewidth=3)
plt.axis('tight')
plt.ylim(-1,10)
ax.axhline(0,color='k')
ax.tick_params(axis='both', which='major', labelsize=15)

#subtract the baseline fit
ax = plt.subplot(212)
plt.plot(new_spec1['Vlsr'], new_spec1['LCP'] - basefit, linewidth=3)
plt.axis('tight')
plt.ylim(-1,10)
plt.ylim(-1,10)
ax.axhline(0,color='k')
ax.tick_params(axis='both', which='major', labelsize=15)

def flatten_baseline(Vlsr, Amp, line_range):
    """
    Remove residual baseline variation after frequency- or position-switching.
    Inputs:
        Vlsr: array of velocity values
        Amp: array of intensity as a function of velocity
        line_range: two-element array with start and end velocity of line emission.
        
    The line emmission range is masked before a fit is performed on the baseline.
    
    Returns: Corrected amplitude
    
    To do:  generalise mask to use multiple line ranges.
    """
    line_chans = np.where(np.logical_and(Vlsr>=line_range[0], Vlsr<=line_range[1]))
    mask = np.zeros(len(Amp))
    mask[line_chans] = 1
    vel_mask = np.ma.array(Vlsr, mask = mask)
    amp_mask = np.ma.array(Amp, mask = mask)

    #In general, we have found that a third-order polynomial is sufficient to fit the spectral baseline
    p = np.poly1d(np.ma.polyfit(vel_mask, amp_mask,  3))
    basefit = p(Vlsr)

    plt.figure(figsize = [15,5])
    plt.plot(Vlsr, Amp)
    plt.plot(vel_mask, amp_mask)
    plt.plot(Vlsr, basefit)
    lim = plt.axis('tight')
    plt.ylim(-2,5)
    plt.ylabel('Amplitude')
    plt.xlabel('Vlsr')

    Amp = Amp - basefit
    
    return Amp

new_spec1['LCP'] = flatten_baseline(new_spec1['Vlsr'], new_spec1['LCP'], line_range)

#repeat for RCP
new_spec1['RCP'] = flatten_baseline(new_spec1['Vlsr'], new_spec1['RCP'], line_range)

new_spec2['LCP'] = flatten_baseline(new_spec2['Vlsr'], new_spec2['LCP'], line_range)
new_spec2['RCP'] = flatten_baseline(new_spec2['Vlsr'], new_spec2['RCP'], line_range)

def ave_spec(spec1, spec2):
    """
    Average two spectra in time.
    Return a new spectrum.
    """
    #find mean of the corrected spectra
    mean_lcp = np.mean(np.vstack([spec1['LCP'], spec2['LCP']]), axis=0)
    mean_rcp = np.mean(np.vstack([spec1['RCP'], spec2['RCP']]), axis=0)
    
    #find mid point between times of observation
    time1 = Time(spec1['date'], scale='utc', format='isot')
    time2 = Time(spec2['date'], scale='utc', format='isot')
    dt = (time2 - time1)/2
    mid_time = time1 +dt
    
    #construct new spectrum
    mean_spec = {'object' : spec1['object'],
                 'longitude' : spec1['longitude'],
                 'latitude' : spec1['latitude'],
                 'equinox' : spec1['equinox'],
                 'HPBW' : spec1['HPBW'],
                 'date' : mid_time.iso,    
                 'Vlsr' : spec1['Vlsr'],
                 'LCP' : mean_lcp,
                 'RCP' : mean_rcp,
                 'Tsys_lcp' : np.mean([spec1['Tsys_lcp'], spec2['Tsys_lcp']]),
                 'DTsys_lcp' : np.mean([spec1['DTsys_lcp'], spec2['DTsys_lcp']]) ,
                 'Tsys_rcp' : np.mean([spec1['Tsys_rcp'], spec2['Tsys_rcp']]),
                 'DTsys_rcp': np.mean([spec1['DTsys_rcp'], spec2['DTsys_rcp']])}
    return mean_spec
    

time_averaged_spec = ave_spec(new_spec1, new_spec2)

plot_spec(time_averaged_spec)





