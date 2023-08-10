from astropy.io import fits

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

#  Set the HSC data file in fits format
fitsfile='/home/babyostrich/data/LSSTDESC/HSC/HSC_WIDE_GAMA15H_forced.fits'

# Open and inspect the file.  This fits file has two blocks, the data is in block 1.
# Refer to astropy documentation: http://docs.astropy.org/en/stable/io/fits/
hdulist = fits.open(fitsfile)

# Check that the 0th block contains data
print "0th block is image: ",hdulist[0].is_image
type(hdulist[1].data)

print hdulist[0].header
print hdulist[1].header

# Let's collect the columns to view, and the data in a table
data_columns = hdulist[1].columns
table_data = hdulist[1].data

# data_columns contains a list of strings labeling the columns
# You should cross-check this with the schema descriptions here: 
#  https://hsc-release.mtk.nao.ac.jp/schema/#:legend
print data_columns.info()

# As a first step, let's plot the ra and dec.  Note, normally we'd need to 
# select out items where ra or dec is null with a mask.  But, there are ra and dec values for
# all items, so the np.where method returns empty arrays.

import numpy as np

#  Check to see that we have all ra and dec for plotting
print np.where(table_data['dec_isnull'])
print np.where(table_data['ra_isnull'])

# Let's plot a subset of the ra and dec to get an idea of the area coverage of this 
# subset of the data
plt.scatter(table_data['dec'][::100], table_data['ra'][::100],s=0.001)
plt.xlabel('dec',fontsize='xx-large')
plt.ylabel('ra',fontsize='xx-large')

# Let's now compare histogram of fluxes and magnitudes depending on how that is calculated. 
# Some values are nan - a quick min/max check on the table column values would show this.

column_name = 'gcmodel_flux'

plt.hist(table_data[column_name][~np.isnan(table_data[column_name])],bins=99)
plt.yscale('log', nonposy='clip')
plt.xlabel('G Flux $[erg s^{-1} cm^{-2} Hz^{-1}]$',fontsize='xx-large')

# Let's check the magnitudes - remove nan and infinite values!
column_label = 'gcmodel_mag'

# Selecting for "np.isfinite" values gets rid of both nan and inf values.
plt.hist(table_data[column_label][np.isfinite(table_data[column_label])],bins=99)
plt.yscale('log', nonposy='clip')
plt.xlabel('G Magnitude',fontsize='xx-large')

# Let's generalize this for all grizy fluxes and magnitudes

def histogram_column_finite_vals(table_data, column_label, annotation) :
    '''Plot a histogram on an axis using a column from the table data.'''
    plt.hist(table_data[column_label][np.isfinite(table_data[column_label])],
             bins=99,normed=True)
    plt.yscale('log',nonposy='clip')
    plt.annotate(annotation, (0.6,0.8), xycoords='figure fraction')

for column_label, annotation in     zip([c+'cmodel_flux' for c in ['g', 'r', 'i', 'z', 'y']],
        [c+' Flux' for c in ['g', 'r', 'i', 'z', 'y']]) :
        histogram_column_finite_vals(table_data, column_label, annotation)
        plt.show()

for column_label, annotation in     zip([c+'cmodel_mag' for c in ['g', 'r', 'i', 'z', 'y']],
        [c+' Magnitude' for c in ['g', 'r', 'i', 'z', 'y']]) :
        histogram_column_finite_vals(table_data, column_label, annotation)
        plt.show()

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

g = 'gcmodel_mag'
r = 'rcmodel_mag'
i = 'icmodel_mag'

r_i_diff = (table_data[r] - table_data[i])
g_r_diff = (table_data[g] - table_data[r])
r_i_diff_fin = r_i_diff[np.isfinite(r_i_diff) & np.isfinite(g_r_diff)]
g_r_diff_fin = g_r_diff[np.isfinite(r_i_diff) & np.isfinite(g_r_diff)]

plt.hist2d(g_r_diff_fin, r_i_diff_fin, bins=99, norm=LogNorm())
plt.colorbar()
plt.xlabel('g-r',fontsize='xx-large')
plt.ylabel('r-i',fontsize='xx-large')

plt.show()



def plot_stargal_diagnostic1(color, table_data) :
    p_c_diff = table_data[color+'mag_psf'] - table_data[color+'cmodel_mag']
    psf_mag = table_data[color+'mag_psf']
    
    p_c_diff_isfin = p_c_diff[np.isfinite(p_c_diff) & np.isfinite(psf_mag)]
    psf_mag_isfin = psf_mag[np.isfinite(p_c_diff) & np.isfinite(psf_mag)]
    
    plt.hist2d(psf_mag_isfin, p_c_diff_isfin, bins=99, norm=LogNorm())
    plt.colorbar()
    plt.xlabel('PSF Mag',fontsize='xx-large')
    plt.ylabel('PSF Mag - cmodel Mag',fontsize='xx-large')
    plt.annotate(color+' band', (0.2,0.8), xycoords='figure fraction',fontsize='large')

    plt.show()

for color in ['g', 'r', 'i', 'z', 'y' ]:
    plot_stargal_diagnostic1(color, table_data)



