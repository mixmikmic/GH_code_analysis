get_ipython().magic('matplotlib inline')
#load modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy import units as u
from astropy.io import fits
from  matplotlib.colors import LogNorm
import scipy.ndimage
#POPPY
import poppy
from poppy.poppy_core import PlaneType
print("POPPY Version: "+poppy.__version__)
#silence DEBUG and WARNINGs
import logging
logging.getLogger('poppy').setLevel(logging.ERROR)
# MagAO-X function definitions
import magaoxFunctions as mf

rx_sys = mf.makeRxCSV('data/rxCSV/rx_magaox_v9.5_PSDset4.csv')

mirrors_array = []
optics_names = []
for t_optic, test_opt in enumerate(rx_sys):
    if test_opt['Type'] == 'mirror' or test_opt['Type'] == 'ref':
        mirrors_array.append(test_opt['Optical_Element_Number'])
        optics_names.append(test_opt['Name'])
        if test_opt['Type'] == 'ref':
            ref_num = test_opt['Optical_Element_Number']
    elif test_opt['Name'] == 'Tweeter':
        tweeter_num = test_opt['Optical_Element_Number']
    elif test_opt['Name'] == 'vAPP-trans':
        vappTRANS_num = test_opt['Optical_Element_Number']
    elif test_opt['Name'] == 'vAPP-opd':
        vappOPD_num = test_opt['Optical_Element_Number']
    
    if test_opt['Name'] == 'M1':
        M1_num = test_opt['Optical_Element_Number']
        
telescope = mirrors_array

# Global terms
# note: u.m is in meters
testWavelength = 656e-9 * u.m # H-alpha
npix = 256  # sampling
beam_ratio = 0.125 # 8x oversample
M1_radius=rx_sys['Radius_m'][M1_num]*u.m

# vAPP files
vAPP_diam = 0.009 # units: meters, always 9mm diameter beam.
vAPP_pixelscl = vAPP_diam/npix # vAPP is sized to whatever the sampling is set.
vAPP_folder = 'data/coronagraph/'
vAPP_trans_filename = 'vAPP_trans_mask_1PSF_256'
vAPP_opd_filename = 'vAPP_opd_mask_1PSF_256'

# Lyot plane pupil file
lyot_pupil_base = fits.open('data/PSDset2/MagAOX_f11_pupil_256_unmasked.fits')[0].data

# Spatial Frequency mapping routine
magKmap = mf.SpatFreqMap(M1_radius, npix)

# Tweeter DM control bandwidth
tweeter_BW = 24/(M1_radius.value*2)

# Tweeter DM diameter - from MagAO-X design: 19.2 x 18.8mm; use the smaller one
tweeter_diam = 0.0188 * u.m

tweeter_pixscl = tweeter_diam.value/npix

get_ipython().run_cell_magic('prun', '', "# Version: 5x DM iteration\n\n# File address locations\ntweeter_surf_folder = 'data/v9.5/s256os8x/set4_g0.4/'\nrx_sys['surf_PSD_folder'][tweeter_num] = tweeter_surf_folder\nsci_intensity_folder = 'output/v9.5/s256os8x/set4/WithDM_g0.4/'\nsci_intensity_prefix = 'SciPSF_WithDM_intensity_'\n\n# Constants\nnum_DM_iteration = 5\ngain = 0.4\n\nprint('Surface check using sampling = %f px and oversampling = %f' % (npix, 1/beam_ratio))\nprint('Case: With DM')\nprint('*****')\n\nfor test_optic in telescope:\n    \n    # Remove test optic PSD and hold the value to change at loop end\n    hold_test_PSD = rx_sys['surf_PSD_filename'][test_optic]\n    rx_sys['surf_PSD_filename'][test_optic] = 'none'\n    \n    # start with no DM surface applied\n    tweeter_surf_prefix = 'DMsurf_' + rx_sys['Name'][test_optic] + '_'\n    tweeter_surf_addr = tweeter_surf_folder + tweeter_surf_prefix\n    rx_sys['surf_PSD_filename'][tweeter_num] = 'none'\n    \n    # DM iteration looping, 5 iterations\n    for DM_count in range(0, num_DM_iteration):\n\n        # build to Lyot plane\n        magaox = mf.csvFresnel(rx_sys, npix, beam_ratio, 'lyot')\n        lyot_psf, lyot_wfs = magaox.calcPSF(wavelength=testWavelength,\n                                    display_intermediates=False,\n                                    return_intermediates=True)\n        lyot_phase_data = lyot_wfs[len(lyot_wfs)-1].asFITS('phase')[0].data\n        \n        # Calculate DM surface based on Lyot phase data\n        calc_surf = mf.BuildLyotDMSurf(lyot_phase_data, npix, lyot_pupil_base, magKmap, tweeter_BW, \n                       testWavelength)\n\n        # Get previous DM surface\n        if DM_count == 0: # First run through\n            DM_surf_prev = np.zeros((npix,npix))\n        else:\n            DM_surf_prev = fits.open(tweeter_surf_addr+str(DM_count-1)+'.fits')[0].data\n        \n        # metric to see surface change\n        DM_metric = np.sum(np.absolute(np.nan_to_num(calc_surf/DM_surf_prev)))\n        #DM_metric_list.append(DM_metric)\n        print('DM iteration %s metric: %f' % (str(DM_count), DM_metric))\n        \n        # If this is the last iteration, then save the calculated surface as the surface for vAPP PSF\n        if DM_count == (num_DM_iteration-1):\n            # Set up next DM surface for full system PSF\n            tweeter_loc = tweeter_surf_prefix + 'final'\n            mf.writeOPDfile(calc_surf, tweeter_pixscl, tweeter_surf_folder + tweeter_loc + '.fits')\n            rx_sys['surf_PSD_filename'][tweeter_num] = tweeter_loc\n        \n        else:\n            # New DM surface to apply\n            DM_new_surf = DM_surf_prev + (gain*calc_surf)\n            mf.writeOPDfile(DM_new_surf, tweeter_pixscl, tweeter_surf_addr + str(DM_count) + '.fits')\n\n            # Set up next DM surface for the next DM iteration\n            rx_sys['surf_PSD_filename'][tweeter_num] = tweeter_surf_prefix + str(DM_count)\n    \n    # Insert vAPP transmission and OPD optics overrides\n    rx_sys['surf_PSD_filename'][vappTRANS_num] = vAPP_trans_filename\n    rx_sys['surf_PSD_filename'][vappOPD_num] = vAPP_opd_filename\n    \n    # Build science plane PSF with DM and vAPP \n    magaox = mf.csvFresnel(rx_sys, npix, beam_ratio, 'sci')   \n    sci_psf = magaox.calcPSF(wavelength=testWavelength, \n                             display_intermediates=False,\n                             return_intermediates=False)\n    sci_intensity_loc = sci_intensity_folder + sci_intensity_prefix + rx_sys['Name'][test_optic] + '.fits'\n    sci_psf.writeto(sci_intensity_loc, clobber=True)\n    \n    # reset corrected optical surfaces\n    rx_sys['surf_PSD_filename'][test_optic] = hold_test_PSD\n    rx_sys['surf_PSD_filename'][tweeter_num] = 'none'\n    rx_sys['surf_PSD_filename'][vappTRANS_num] = 'none'\n    rx_sys['surf_PSD_filename'][vappOPD_num] = 'none'\n\n    print('%s surface test completed' % rx_sys['Name'][test_optic])")

get_ipython().run_cell_magic('prun', '', "# Version: 5x DM iteration, different \n\n# Open new CSV file\nrx_sys = mf.makeRxCSV('data/rxCSV/rx_magaox_v9.5_PSDset4.csv')\n\n# File address locations\ntweeter_surf_folder = 'data/v9.5/s256os8x/set4_g0.9/'\nrx_sys['surf_PSD_folder'][tweeter_num] = tweeter_surf_folder\nsci_intensity_folder = 'output/v9.5/s256os8x/set4/WithDM_g0.9/'\nsci_intensity_prefix = 'SciPSF_WithDM_intensity_'\n\n# Constants\nnum_DM_iteration = 5\ngain = 0.9\n\nprint('Surface check using sampling = %f px and oversampling = %f' % (npix, 1/beam_ratio))\nprint('Case: With DM')\nprint('*****')\n\nfor test_optic in telescope:\n    \n    # Remove test optic PSD and hold the value to change at loop end\n    hold_test_PSD = rx_sys['surf_PSD_filename'][test_optic]\n    rx_sys['surf_PSD_filename'][test_optic] = 'none'\n    \n    # start with no DM surface applied\n    tweeter_surf_prefix = 'DMsurf_' + rx_sys['Name'][test_optic] + '_'\n    tweeter_surf_addr = tweeter_surf_folder + tweeter_surf_prefix\n    rx_sys['surf_PSD_filename'][tweeter_num] = 'none'\n    \n    # DM iteration looping, 5 iterations\n    for DM_count in range(0, num_DM_iteration):\n\n        # build to Lyot plane\n        magaox = mf.csvFresnel(rx_sys, npix, beam_ratio, 'lyot')\n        lyot_psf, lyot_wfs = magaox.calcPSF(wavelength=testWavelength,\n                                    display_intermediates=False,\n                                    return_intermediates=True)\n        lyot_phase_data = lyot_wfs[len(lyot_wfs)-1].asFITS('phase')[0].data\n        \n        # Calculate DM surface based on Lyot phase data\n        calc_surf = mf.BuildLyotDMSurf(lyot_phase_data, npix, lyot_pupil_base, magKmap, tweeter_BW, \n                       testWavelength)\n\n        # Get previous DM surface\n        if DM_count == 0: # First run through\n            DM_surf_prev = np.zeros((npix,npix))\n        else:\n            DM_surf_prev = fits.open(tweeter_surf_addr+str(DM_count-1)+'.fits')[0].data\n        \n        # metric to see surface change\n        DM_metric = np.sum(np.absolute(np.nan_to_num(calc_surf/DM_surf_prev)))\n        #DM_metric_list.append(DM_metric)\n        print('DM iteration %s metric: %f' % (str(DM_count), DM_metric))\n        \n        # If this is the last iteration, then save the calculated surface as the surface for vAPP PSF\n        if DM_count == (num_DM_iteration-1):\n            # Set up next DM surface for full system PSF\n            tweeter_loc = tweeter_surf_prefix + 'final'\n            mf.writeOPDfile(calc_surf, tweeter_pixscl, tweeter_surf_folder + tweeter_loc + '.fits')\n            rx_sys['surf_PSD_filename'][tweeter_num] = tweeter_loc\n        \n        else:\n            # New DM surface to apply\n            DM_new_surf = DM_surf_prev + (gain*calc_surf)\n            mf.writeOPDfile(DM_new_surf, tweeter_pixscl, tweeter_surf_addr + str(DM_count) + '.fits')\n\n            # Set up next DM surface for the next DM iteration\n            rx_sys['surf_PSD_filename'][tweeter_num] = tweeter_surf_prefix + str(DM_count)\n    \n    # Insert vAPP transmission and OPD optics overrides\n    rx_sys['surf_PSD_filename'][vappTRANS_num] = vAPP_trans_filename\n    rx_sys['surf_PSD_filename'][vappOPD_num] = vAPP_opd_filename\n    \n    # Build science plane PSF with DM and vAPP \n    magaox = mf.csvFresnel(rx_sys, npix, beam_ratio, 'sci')   \n    sci_psf = magaox.calcPSF(wavelength=testWavelength, \n                             display_intermediates=False,\n                             return_intermediates=False)\n    sci_intensity_loc = sci_intensity_folder + sci_intensity_prefix + rx_sys['Name'][test_optic] + '.fits'\n    sci_psf.writeto(sci_intensity_loc, clobber=True)\n    \n    # reset corrected optical surfaces\n    rx_sys['surf_PSD_filename'][test_optic] = hold_test_PSD\n    rx_sys['surf_PSD_filename'][tweeter_num] = 'none'\n    rx_sys['surf_PSD_filename'][vappTRANS_num] = 'none'\n    rx_sys['surf_PSD_filename'][vappOPD_num] = 'none'\n\n    print('%s surface test completed' % rx_sys['Name'][test_optic])")

get_ipython().run_cell_magic('prun', '', "# Version: 1x DM iteration\n\n# File address locations\ntweeter_surf_folder = 'data/v9.5/s256os8x/set5/'\nrx_sys['surf_PSD_folder'][tweeter_num] = tweeter_surf_folder\nsci_intensity_folder = 'output/v9.5/s256os8x/set5/WithDM/'\nsci_intensity_prefix = 'SciPSF_WithDM_intensity_'\n\nprint('Surface check using sampling = %f px and oversampling = %f' % (npix, 1/beam_ratio))\nprint('Case: With DM, single DM iteration, no gain implemented')\nprint('*****')\n\nfor test_optic in telescope:\n    \n    # Remove test optic PSD and hold the value to change at loop end\n    hold_test_PSD = rx_sys['surf_PSD_filename'][test_optic]\n    rx_sys['surf_PSD_filename'][test_optic] = 'none'\n\n    # build to Lyot plane\n    magaox = mf.csvFresnel(rx_sys, npix, beam_ratio, 'lyot')\n    lyot_psf, lyot_wfs = magaox.calcPSF(wavelength=testWavelength,\n                                display_intermediates=False,\n                                return_intermediates=True)\n    lyot_phase_data = lyot_wfs[len(lyot_wfs)-1].asFITS('phase')[0].data\n\n    # Calculate DM surface based on Lyot phase data\n    calc_surf = mf.BuildLyotDMSurf(lyot_phase_data, npix, lyot_pupil_base, magKmap, tweeter_BW, \n                   testWavelength)\n    tweeter_surf_file_loc = tweeter_surf_folder + 'DMsurf_' + rx_sys['Name'][test_optic] + '.fits'\n    mf.writeOPDfile(calc_surf, tweeter_pixscl, tweeter_surf_file_loc)\n    \n    # Insert vAPP transmission and OPD optics overrides\n    rx_sys['surf_PSD_filename'][tweeter_num] = 'DMsurf_' + rx_sys['Name'][test_optic]\n    rx_sys['surf_PSD_filename'][vappTRANS_num] = vAPP_trans_filename\n    rx_sys['surf_PSD_filename'][vappOPD_num] = vAPP_opd_filename\n    \n    # Build science plane PSF with DM and vAPP \n    magaox = mf.csvFresnel(rx_sys, npix, beam_ratio, 'sci')   \n    sci_psf = magaox.calcPSF(wavelength=testWavelength, \n                             display_intermediates=False,\n                             return_intermediates=False)\n    sci_intensity_loc = sci_intensity_folder + sci_intensity_prefix + rx_sys['Name'][test_optic] + '.fits'\n    sci_psf.writeto(sci_intensity_loc, clobber=True)\n    \n    # reset corrected optical surfaces\n    rx_sys['surf_PSD_filename'][test_optic] = hold_test_PSD\n    rx_sys['surf_PSD_filename'][tweeter_num] = 'none'\n    rx_sys['surf_PSD_filename'][vappTRANS_num] = 'none'\n    rx_sys['surf_PSD_filename'][vappOPD_num] = 'none'\n\n    print('%s surface test completed' % rx_sys['Name'][test_optic])")

# Dark Hole geometry from ds9, for 256pix sample at 8x oversample setup
center_pix = [1023.5, 970.5] # changed for 54x54 size region
shift=26.5 # hardcoded for 54x54 size region

# calculated from a previous iteration
set1_PSF_withDM_loc = 'output/v9.5/s256os8x/set1/WithDM/SciPSF_WithDM_intensity_'
set1_PSF_withoutDM_loc = 'output/v9.5/s256os8x/set1/WithoutDM/SciPSF_WithoutDM_intensity_'
set1_PSF_withDM_5x_loc = 'output/v9.5/s256os8x/set1/WithDM_g0.9/SciPSF_WithDM_intensity_'
set1_withDM_avg, set1_withDM_avg_ref = mf.calcFluxDH_List(set1_PSF_withDM_loc, center_pix, shift, 
                                                          optics_names, 'mean')
set1_withoutDM_avg, set1_withoutDM_avg_ref = mf.calcFluxDH_List(set1_PSF_withoutDM_loc, center_pix, shift, 
                                                          optics_names, 'mean')
set1_withDM_5x_avg, set1_withDM_5x_avg_ref = mf.calcFluxDH_List(set1_PSF_withDM_5x_loc, center_pix, shift, 
                                                          optics_names, 'mean')

set2_PSF_withDM_loc = 'output/v9.5/s256os8x/set2/WithDM/SciPSF_WithDM_intensity_'
set2_PSF_withoutDM_loc = 'output/v9.5/s256os8x/set2/WithoutDM/SciPSF_WithoutDM_intensity_'
set2_withDM_avg, set2_withDM_avg_ref = mf.calcFluxDH_List(set2_PSF_withDM_loc, center_pix, shift, 
                                                          optics_names, 'mean')
set2_withoutDM_avg, set2_withoutDM_avg_ref = mf.calcFluxDH_List(set2_PSF_withoutDM_loc, center_pix, shift, 
                                                          optics_names, 'mean')

set3_PSF_withDM_loc = 'output/v9.5/s256os8x/set3/WithDM/SciPSF_WithDM_intensity_'
set3_withDM_avg, set3_withDM_avg_ref = mf.calcFluxDH_List(set3_PSF_withDM_loc, center_pix, shift, 
                                                          optics_names, 'mean')

set4_PSF_withDM_loc = 'output/v9.5/s256os8x/set4/WithDM/SciPSF_WithDM_intensity_'
set4_PSF_withDM_5x_g09_loc = 'output/v9.5/s256os8x/set4/WithDM_g0.9/SciPSF_WithDM_intensity_'
set4_PSF_withDM_5x_g04_loc = 'output/v9.5/s256os8x/set4/WithDM_g0.4/SciPSF_WithDM_intensity_'
set4_withDM_avg, set4_withDM_avg_ref = mf.calcFluxDH_List(set4_PSF_withDM_loc, center_pix, shift, 
                                                          optics_names, 'mean')
set4_withDM_5x_g09_avg, set4_withDM_5x_g09_avg_ref = mf.calcFluxDH_List(set4_PSF_withDM_5x_g09_loc, center_pix, shift, 
                                                          optics_names, 'mean')
set4_withDM_5x_g04_avg, set4_withDM_5x_g04_avg_ref = mf.calcFluxDH_List(set4_PSF_withDM_5x_g04_loc, center_pix, shift, 
                                                          optics_names, 'mean')

set5_PSF_withDM_loc = 'output/v9.5/s256os8x/set5/WithDM/SciPSF_WithDM_intensity_'
set5_withDM_avg, set5_withDM_avg_ref = mf.calcFluxDH_List(set5_PSF_withDM_loc, center_pix, shift, 
                                                          optics_names, 'mean')

x=np.arange(len(optics_names))
plt.plot(x, set4_withDM_avg, 'bo-', label='1x DM')
plt.plot(x, set4_withDM_5x_g09_avg, 'go-', label='5x DM, g=0.9')
plt.plot(x, set4_withDM_5x_g04_avg, 'ro-', label='5x DM, g=0.4')
plt.axhline(y=set4_withDM_avg_ref, color='b', linestyle='-')
plt.axhline(y=set4_withDM_5x_g09_avg_ref, color='g', linestyle='-')
plt.axhline(y=set4_withDM_5x_g04_avg_ref, color='r', linestyle='-')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xticks(x,optics_names,rotation='vertical')
plt.margins(0.1)
plt.subplots_adjust(top=1.0)
plt.subplots_adjust(bottom=0.15)
plt.title('Average Contrast in DH With DM, PSD Set 4',y=1.05)
plt.show()

x=np.arange(len(optics_names))
plt.plot(x, set1_withDM_avg, 'ro-', label='PSD Set 1')
plt.plot(x, set2_withDM_avg, 'mo-', label='PSD Set 2')
plt.plot(x, set3_withDM_avg, 'go-', label='PSD Set 3')
plt.plot(x, set4_withDM_avg, 'bo-', label='PSD Set 4 (pixscale match)')
plt.plot(x, set5_withDM_avg, 'ko-', label='PSD Set 5 (pixscale match)')
plt.axhline(y=set1_withDM_avg_ref, color='r', linestyle='-')
plt.axhline(y=set2_withDM_avg_ref, color='m', linestyle='-')
plt.axhline(y=set3_withDM_avg_ref, color='g', linestyle='-')
plt.axhline(y=set4_withDM_avg_ref, color='b', linestyle='-')
plt.axhline(y=set5_withDM_avg_ref, color='k', linestyle='-')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xticks(x,optics_names,rotation='vertical')
plt.margins(0.1)
plt.subplots_adjust(top=1.0)
plt.subplots_adjust(bottom=0.15)
plt.title('Average Contrast in DH (With DM, 1x iteration)',y=1.05)
plt.show()

x=np.arange(len(optics_names))
plt.plot(x, set4_withDM_avg, 'bo-', label='Set 4')
plt.plot(x, set5_withDM_avg, 'ko-', label='Set 5')
plt.axhline(y=set4_withDM_avg_ref, color='b', linestyle='-')
plt.axhline(y=set5_withDM_avg_ref, color='k', linestyle='-')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xticks(x,optics_names,rotation='vertical')
plt.margins(0.1)
plt.subplots_adjust(top=1.0)
plt.subplots_adjust(bottom=0.15)
plt.title('Average Contrast in DH With 1-DM, PSD Set 4,5',y=1.05)
plt.show()

x=np.arange(len(optics_names))
plt.plot(x, set1_withDM_avg, 'ro-', label='1x DM iteration')
plt.plot(x, set1_withDM_5x_avg, 'mo-', label='5x DM iteration')
plt.axhline(y=set1_withDM_avg_ref, color='r', linestyle='-')
plt.axhline(y=set1_withDM_5x_avg_ref, color='m', linestyle='-')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xticks(x,optics_names,rotation='vertical')
plt.margins(0.1)
plt.subplots_adjust(top=1.0)
plt.subplots_adjust(bottom=0.15)
plt.title('Average Contrast in DH With DM, PSD Set 1',y=1.05)
plt.show()

get_ipython().run_cell_magic('prun', '', '# File address locations\nsci_intensity_folder = \'output/v9.5/s256os8x/set1/WithoutDM/\'\nsci_intensity_prefix = \'SciPSF_WithoutDM_intensity_\'\n\n# Insert vAPP transmission and OPD optics overrides\nprint(\'Override vAPP OPD and transmission PSD maps for WithoutDM case\')\nrx_sys[\'surf_PSD_filename\'][vappTRANS_num] = vAPP_trans_filename\nrx_sys[\'surf_PSD_filename\'][vappOPD_num] = vAPP_opd_filename\n\n# Make sure there is no surface map on tweeter DM\nrx_sys[\'surf_PSD_filename\'][tweeter_num] = \'none\'\n\nprint(\'Surface check (Without DM) using sampling = %f px and oversampling = %f\' % (npix, 1/beam_ratio))\nprint(\'*****\')\n\nfor test_optic in telescope:\n    print(\'Build MagAO-X removing %s surface without DM\' % rx_sys[\'Name\'][test_optic])\n    \n    # Remove test optic PSD and hold the value to change at loop end\n    hold_test_PSD = rx_sys[\'surf_PSD_filename\'][test_optic]\n    rx_sys[\'surf_PSD_filename\'][test_optic] = \'none\'\n    \n    # Build science plane PSF with DM and vAPP \n    magaox = mf.csvFresnel(rx_sys, npix, beam_ratio, \'sci\')   \n    sci_psf = magaox.calcPSF(wavelength=testWavelength, \n                             display_intermediates=False,\n                             return_intermediates=False)\n    sci_intensity_loc = sci_intensity_folder + sci_intensity_prefix + rx_sys[\'Name\'][test_optic] + \'.fits\'\n    sci_psf.writeto(sci_intensity_loc, clobber=True)\n    \n    # reset corrected optical surfaces\n    rx_sys[\'surf_PSD_filename\'][test_optic] = hold_test_PSD\n\n    print(\'End test for %s surface (without DM)\' % rx_sys[\'Name\'][test_optic])\n    print(\'*****\')\n    \n# Reset rx_sys filenames for vAPP to look like original setup\nprint(\'Reset vAPP OPD and transmission PSD maps back to "none"\')\nrx_sys[\'surf_PSD_filename\'][vappTRANS_num] = \'none\'\nrx_sys[\'surf_PSD_filename\'][vappOPD_num] = \'none\'')

vapp_phase_data = fits.open('data/coronagraph/MagAOx_256x256_phase_nograting.fits')[0].data
vapp_aperture_data = fits.open('data/coronagraph/MagAOx_256x256_amp.fits')[0].data

vAPP_folder = 'data/coronagraph/'

#vapp_opd_surf_data = 1*(testWavelength.value/(2*np.pi))*vapp_phase_data*vapp_aperture_data
vapp_opd_surf_data = 10*(testWavelength.value/(2*np.pi))*vapp_phase_data*vapp_aperture_data
vAPP_opd_filename = 'vAPP_opd_mask_1PSF_256_x10'
mf.writeOPDfile(vapp_opd_surf_data, vAPP_pixelscl, vAPP_folder + vAPP_opd_filename + '.fits')

vAPP_trans_filename = 'vAPP_trans_mask_1PSF_256'
mf.writeTRANSfile(vapp_aperture_data, vAPP_pixelscl, vAPP_folder + vAPP_trans_filename + '.fits')

vapp_phase_data = fits.open('data/coronagraph/MagAOx_256x256_phase_grating.fits')[0].data
vapp_aperture_data = fits.open('data/coronagraph/MagAOx_256x256_amp.fits')[0].data

vAPP_folder = 'data/coronagraph/'

vapp_opd_surf_data = 1*(testWavelength.value/(2*np.pi))*vapp_phase_data*vapp_aperture_data
vAPP_opd_filename_2PSF = 'vAPP_opd_mask_2PSF_256'
mf.vappOPDfile(vapp_opd_surf_data, vAPP_pixelscl, vAPP_folder + vAPP_opd_filename_2PSF + '.fits')

#vAPP_trans_filename = 'vAPP_trans_mask_1PSF_256'
#mf.vappTRANSfile(vapp_aperture_data, vAPP_pixelscl, vAPP_folder + vAPP_trans_filename + '.fits')

# File address locations
tweeter_surf_folder = 'data/v9.5/s256os8x/set1_2PSF/'
rx_sys['surf_PSD_folder'][tweeter_num] = tweeter_surf_folder
tweeter_surf_prefix = 'DMsurf_'
sci_intensity_folder = 'output/v9.5/s256os8x/set1_2PSF/WithDM/'
sci_intensity_prefix = 'SciPSF_WithDM_intensity_'

print('Surface check using sampling = %f px and oversampling = %f' % (npix, 1/beam_ratio))
print('*****')

telescope=[ref_num]

for test_optic in telescope:
    print('Build MagAO-X removing surface: %s' % rx_sys['Name'][test_optic])
    
    # Remove test optic PSD and hold the value to change at loop end
    hold_test_PSD = rx_sys['surf_PSD_filename'][test_optic]
    rx_sys['surf_PSD_filename'][test_optic] = 'none'
    
    # build to Lyot plane
    magaox = mf.csvFresnel(rx_sys, npix, beam_ratio, 'lyot')
    lyot_psf, lyot_wfs = magaox.calcPSF(wavelength=testWavelength,
                                    display_intermediates=False,
                                    return_intermediates=True)
    lyot_phase_plane_data = lyot_wfs[len(lyot_wfs)-1].asFITS('phase')[0].data
    
    # Build Lyot DM surface map routine; spatial plane must be built prior to for-loop
    tweeter_surf_file_loc = tweeter_surf_folder + tweeter_surf_prefix + rx_sys['Name'][test_optic]
    mf.BuildLyotDMSurf(lyot_phase_plane_data, npix, tweeter_diam, lyot_pupil_base, magKmap, tweeter_BW, 
                       testWavelength, tweeter_surf_file_loc)
    
    # Insert Lyot DM map vAPP transmission and OPD optics overrides
    rx_sys['surf_PSD_filename'][tweeter_num] = tweeter_surf_prefix + rx_sys['Name'][test_optic]
    rx_sys['surf_PSD_filename'][vappTRANS_num] = vAPP_trans_filename
    rx_sys['surf_PSD_filename'][vappOPD_num] = vAPP_opd_filename_2PSF
    
    # Build science plane PSF with DM and vAPP 
    magaox = mf.csvFresnel(rx_sys, npix, beam_ratio, 'sci')   
    sci_psf = magaox.calcPSF(wavelength=testWavelength, 
                             display_intermediates=False,
                             return_intermediates=False)
    sci_intensity_loc = sci_intensity_folder + sci_intensity_prefix + rx_sys['Name'][test_optic] + '.fits'
    sci_psf.writeto(sci_intensity_loc, clobber=True)
    
    # reset corrected optical surfaces
    rx_sys['surf_PSD_filename'][test_optic] = hold_test_PSD
    rx_sys['surf_PSD_filename'][tweeter_num] = 'none'
    rx_sys['surf_PSD_filename'][vappTRANS_num] = 'none'
    rx_sys['surf_PSD_filename'][vappOPD_num] = 'none'

    print('End test for %s surface' % rx_sys['Name'][test_optic])
    print('*****')

