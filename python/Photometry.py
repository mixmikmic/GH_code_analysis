import photutils
from photutils import detect_sources, detect_threshold, aperture_photometry
from photutils import source_properties, properties_table
from photutils import Background2D, MedianBackground
from photutils import CircularAnnulus, CircularAperture
from photutils.utils import random_cmap
from photutils.utils import calc_total_error

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy import wcs
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import hstack
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')

extinction_coeffs = {'U': 0.50,'B': 0.25, 'V': 0.15, 'R': 0.10, 'I': 0.07}

def flux_airmass_correction(flux, airmass, filt):
    """This function corrects the measured flux to an airmass of 0"""
    
    return flux * np.exp(0.4 * extinction_coeffs[filt] * airmass)

# Science data
data_dir = 'Sampledata_Astro501/'
img_file = data_dir + '0121_grb030329_reduced_astrometry.fits'
data, header = fits.getdata(img_file, header=True)
data = np.nan_to_num(data)  # Make any NaN's = 0

# The MDM keyword for airmass is SECZ
# The MDM keyword for the filter is MISFLTID
# Make sure you adjust the next couple of lines!

# airmass = header['SECZ']
# filt = header['MISFLTID']
airmass = header['AIRMASS']
filt = 'B'

wcs_tranform = wcs.WCS(header)

plt.imshow(data, origin='lower', cmap='gray', vmin=260, vmax=400)
plt.colorbar()
plt.tight_layout()

# Standard data

# CHANGE THIS IF STANDARD AND SCIENCE OBJECTS ARE IN DIFFERENT IMAGES
standard_img_file = data_dir + '0121_grb030329_reduced_astrometry.fits'
###

standard_data, standard_header = fits.getdata(standard_img_file, header=True)
standard_data = np.nan_to_num(standard_data)  # Make any NaN's = 0

# The MDM keyword for airmass is SECZ
# The MDM keyword for the filter is MISFLTID
# Make sure you adjust the next couple of lines!

# standard_airmass = standard_header['SECZ']
# standard_filt = standard_header['MISFLTID']
standard_airmass = standard_header['AIRMASS']
standard_filt = 'B'

standard_wcs_tranform = wcs.WCS(standard_header)

plt.imshow(standard_data, origin='lower', cmap='gray', vmin=260, vmax=400)
plt.colorbar()
plt.tight_layout()

obj_coords_table = pd.read_csv(data_dir+'object_coords.cat', delimiter=',')
obj_coords = SkyCoord(obj_coords_table.ra, obj_coords_table.dec, 
                      unit=(u.hour, u.deg), equinox=obj_coords_table.epoch.astype(str))

# Convert sky coordinates to pixels using the WCS
x_pix, y_pix = obj_coords.to_pixel(wcs_tranform)
pix_coords = list(zip(x_pix, y_pix))


plt.imshow(data, origin='lower', cmap='gray', vmin=260, vmax=400)
plt.scatter(x_pix[0], y_pix[0], marker='*', color='r', label='GRB')
plt.scatter(x_pix[1], y_pix[1], marker='x', color='c', label='Standard')
plt.legend()
plt.tight_layout()

# Plot photon counts as a function of radius given the position
# the source. Useful for determining aperture parameters
def plot_radial_counts(data, source_centroids):
    n_sources = len(source_centroids)
    fig, axes = plt.subplots(n_sources, 1, 
                             sharex=True, figsize=(6, 3*n_sources))

    # Loop over sources
    for i in range(n_sources):
        radius = 0
        radii = []
        photon_counts = []
        
        # Most sources are much smaller than 25 pixels in radius
        # But can definitely increase this value if needed!
        # This also gives a good sense of background counts
        while radius < 25:
            # Create the aperture
            aperture = CircularAnnulus(source_centroids[i],
                                       r_in = radius, r_out = radius+1)
            # Do the photometry
            phot_table = aperture_photometry(data, aperture, method='subpixel')
            
            # Determine number of photon counts per unit area
            photon_counts.append(phot_table['aperture_sum']/aperture.area())

            radii.append(radius)
            radius += 1
            
        # Plot the data
        # plt.clf()
        axes[i].grid(True)
        axes[i].plot(radii, photon_counts)
        axes[i].set_ylabel("ADU Counts per Unit Area")
        axes[i].text(0.9,0.9, obj_coords_table.name[i], horizontalalignment='center',
                     verticalalignment='center',transform=axes[i].transAxes, 
                     bbox=dict(facecolor='red', alpha=0.2))
        
    fig.subplots_adjust(hspace=0.05)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    plt.suptitle("ADU counts vs. Radial Distance")
    plt.xlabel("Radial Distance (pixels)")
    # plt.ylabel("ADU Counts per Unit Area")
        
plot_radial_counts(data, pix_coords)

# Define your aperture sizes (in pixels)
# Add additional apertures if you have more than two objects
standard_ap, standard_bkg_in, standard_bkg_out = 7, 12, 17
obj_aps, obj_bkgs_in, obj_bkgs_out = 7, 15, 20

def do_ap_photometry(data, source_centroid, r_ap, bg_rin, bg_rout, airmass, filt):
    """
    Args:
        data:            The image data array
        source_centroid: Location of source (pix coordinates)
        r_ap:            Size (in pix) of source aperture
        bg_rin:          Inner radius (in pix) of background annulus
        bg_rout:         Outer radius (in pix) of background annulus
        airmass:         Airmass of the image
        filt:            Filter used for the image
        
    Returns:
        Flux and flux error due to source, background subtracted
    """
    
    # Change to appropriate MDM values
    # Values are likely in FITS header...
    read_noise = 10
    gain = 4
    
    # Create the source and background apertures
    aperture = CircularAperture(source_centroid, r=r_ap)
    bg_aperture = CircularAnnulus(source_centroid,
                                  r_in = bg_rin,
                                  r_out = bg_rout)

    # Do the aperture photometry                              
    rawflux_table = aperture_photometry(data, aperture)
    bkgflux_table = aperture_photometry(data, bg_aperture)

    # Correct for aatmospheric extinction
    rawflux_table['aperture_sum_corr'] = flux_airmass_correction(rawflux_table['aperture_sum'], airmass, filt)
    bkgflux_table['aperture_sum_corr'] = flux_airmass_correction(bkgflux_table['aperture_sum'], airmass, filt)
    
    phot_table = hstack([rawflux_table, bkgflux_table], 
                         table_names=['raw', 'bkg'])

    # Get the mean number of background counts per unit area                    
    bkg_mean = phot_table['aperture_sum_corr_bkg'] / bg_aperture.area()

    # Calculate number of background counts in source aperture
    bkg_sum = bkg_mean*aperture.area()

    # Calculate the true number of source photon counts in
    # source aperture
    final_flux = phot_table["aperture_sum_corr_raw"]-bkg_sum
    final_flux_err = np.sqrt(final_flux + bkg_sum + aperture.area()*read_noise**2*gain)
    
    print ("Background sum = ", bkg_sum.data, "\n")
    print ("Object Counts = ", final_flux.data, "\n")
    
    return final_flux, final_flux_err


def calculate_zeropoint(data, source_centroid, r_ap,
                        bg_rin, bg_rout, known_mag, airmass, filt):
    """
    Args:
        data:            The image data array
        source_centroid: Location of source (pix coordinates)
        r_ap:            Size (in pix) of source aperture
        bg_rin:          Inner radius (in pix) of background annulus
        bg_rout:         Outer radius (in pix) of background annulus
        airmass:         Airmass of the image
        filt:            Filter used for the image
        
    Returns:
        The zero point magnitude and error
    """

    final_flux, final_flux_err = do_ap_photometry(data, source_centroid, r_ap, 
                                                  bg_rin, bg_rout, airmass, filt)

    # Calculate the zero point using the known magnitude
    zp =  known_mag+2.5*np.log10(final_flux)
    zp_err = np.sqrt( ((-2.5/final_flux)*final_flux_err)**2)
    
    return zp.data, zp_err.data


def calculate_magnitude(data, source_centroid, r_ap, 
                        bg_rin, bg_rout, zeropoint, zeropoint_err,
                        airmass, filt):
    """
    Args:
        data:            The image data array
        source_centroid: Location of source (pix coordinates)
        r_ap:            Size (in pix) of source aperture
        bg_rin:          Inner radius (in pix) of background annulus
        bg_rout:         Outer radius (in pix) of background annulus
        airmass:         Airmass of the image
        filt:            Filter used for the image
        
    Returns:
        The object magnitude and error
    """
    
    final_flux, final_flux_err = do_ap_photometry(data, source_centroid, r_ap, 
                                                  bg_rin, bg_rout, airmass, filt)
    
    # Calculate source magnitude using source photon counts
    # and previously determined zero point
    mag = -2.5*np.log10(final_flux) + zeropoint

    mag_err = np.sqrt( ((-2.5/final_flux)*final_flux_err)**2 + zeropoint_err**2)

    return mag.data, mag_err.data

print("-"*70)
print("| Zero point")
print("-"*70)
zeropoint, zeropoint_err = calculate_zeropoint(standard_data, (x_pix[1], y_pix[1]), 
                                               standard_ap, standard_bkg_in, 
                                               standard_bkg_out, obj_coords_table.mag[1],
                                               standard_airmass, standard_filt)
print("Zeropoint, error = ",zeropoint, zeropoint_err)


print("\n" + "-"*70)
print("| Science Object Mag")
print("-"*70)
obj_mag, obj_err = calculate_magnitude(data, (x_pix[0], y_pix[0]), 
                                       obj_aps, obj_bkgs_in, obj_bkgs_out, 
                                       zeropoint, zeropoint_err,
                                       airmass, filt)
print("Magnitude, error = ",obj_mag, obj_err)

# First, determine the 2D background map of your image
bkg_estimator = MedianBackground()
bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                   bkg_estimator=bkg_estimator)
threshold = bkg.background + (3. * bkg.background_rms)

# Remember to change the gain to the correct value for MDM
gain = 4
error = calc_total_error(data, bkg_error=bkg.background_rms, effective_gain=gain)

segmented_img = detect_sources(data, threshold, npixels=5)
props = source_properties(data, segmented_img, 
                          error=error, wcs=wcs_tranform, background=bkg.background)

tbl = properties_table(props)
print(tbl)
# Use print(tbl.columns) to see what data is contained in this table

r = 5.    # approximate isophotal extent, i.e. your best guess for source aperture size
apertures = []
for prop in props:
    position = (prop.xcentroid.value, prop.ycentroid.value)
    r_new = prop.equivalent_radius.value
    apertures.append(CircularAperture(position, r=r_new))


rand_cmap = random_cmap(segmented_img.max + 1, random_state=12345)
norm = ImageNormalize(stretch=SqrtStretch())
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
ax1.imshow(data, origin='lower', cmap='Greys_r', norm=norm)
ax2.imshow(segmented_img, origin='lower', cmap=rand_cmap)
for aperture in apertures:
    aperture.plot(color='blue', lw=1.5, alpha=0.5, ax=ax1)
    aperture.plot(color='white', lw=1.0, alpha=1.0, ax=ax2)

tbl['source_sum_corr'] = flux_airmass_correction(tbl['source_sum'], airmass, filt)
tbl['source_mag'] = -2.5*np.log10(tbl['source_sum_corr']) + zeropoint
tbl['source_mag_err'] = np.sqrt( ((-2.5/tbl['source_sum_corr'])*tbl['source_sum_err'])**2 
                                + zeropoint_err**2
                               )

pos_error = 3/3600. * u.deg  # Allow wiggle room of 3 arcsec
ra_cond = (np.abs(tbl['ra_icrs_centroid'] - obj_coords[1].ra) < pos_error)
dec_cond = (np.abs(tbl['dec_icrs_centroid'] - obj_coords[1].dec) < pos_error)

obj_tbl = tbl[ra_cond & dec_cond]
print(obj_tbl)

