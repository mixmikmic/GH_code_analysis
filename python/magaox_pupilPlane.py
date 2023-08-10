get_ipython().magic('matplotlib inline')
#load modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy import units as u
from astropy.io import fits
from  matplotlib.colors import LogNorm 
#POPPY
import poppy
from poppy.poppy_core import PlaneType
print("POPPY Version: "+poppy.__version__)

# note: u.m is in meters
testWavelength = 656e-9 * u.m # H-alpha
npix = 256  # resolution

# physical radius values
M1_radius = 6.5/2 * u.m # This is the correct value to use.
#M1_radius = 3.2392 * u.m # This is the Zemax inputted value.
M2_radius = 1.339/2 * u.m # This is the correct value to use.
#M2_radius = 0.632839 * u.m # This is the Zemax inputted value
M3_radius = 0.439879 * u.m # Using largest value from various Zemax files
oap_radius = 0.0254 * u.m # 2 inch diameter OAP
flat_radius = 0.025 * u.m # 50mm diameter

# focal lengths
fl_M1 = 8.128 * u.m
fl_ratio_M2 = 1.07
#fl_M2 = M2_diam * fl_ratio_M2
fl_M2 = 1.43273 * u.m # This is the correct value to use.
fl_OAP0 = 0.14355 * u.m # This is the correct value to use, not the Zemax calculated one.

# Math check for the OAPs in the zemax file
oap_roc = np.array([204.431, 337.544, 589.294, 2106.685, 1008.520, 1220.84, 1220.84, 1220.84]) / 1000 
oap_angle = np.array([65, 53, 20, 15, 15, 15, 15, 15])
oap_fl = oap_roc/(1+np.cos(np.deg2rad(oap_angle))) # equation defined in a presentation somewhere

# Primary and Secondary mirrors
M1 = poppy.QuadraticLens(fl_M1, name='M-1')
M2 = poppy.QuadraticLens(fl_M2, name='M-2')

# OAP mirrors
#OAP0 = poppy.QuadraticLens(oap_fl[0] * u.m, name='OAP-0')
OAP0 = poppy.QuadraticLens(fl_OAP0, name='OAP-0')

# propagation distances based on Zemax
d_m1_m2 = 9.72205 * u.m
d_m2_m3 = 9.02279 * u.m
d_m3_wfsdichroic = 4.849516 * u.m
d_wfsdichroic_peri1 = 0.100 * u.m
d_peri1_f11fp = 0.030 * u.m
d_f11fp_peri2 = 0.033204 * u.m # zemax value
#d_peri2_oap0 = 0.010 * u.m # Zemax value. Doesn't look right one bit.
#d_peri2_oap0 = 0.110346 * u.m # Calculated by FL_OAP0 - d_f11fp_peri2
#d_oap0_k1 = 0.081125 * u.m
#d_k1_k2 = 0.025 * u.m
#d_k2_k3 = 0.025 * u.m
#d_k3_woofer = 0.055 * u.m
# Below are all Zemax values.
d_peri2_oap0 = 0.110503 * u.m # Calculated by FL_OAP0 - d_f11fp_peri2
d_oap0_k1 = 0.039639 * u.m
d_k1_k2 = 0.025 * u.m
d_k2_k3 = 0.025 * u.m
d_k3_woofer = 0.055 * u.m

# Saved F/# correction values, calculated from no-aberrations version
f11_delta = 0.15491726 * u.m

def surfFITS(file_loc, optic_type, opdunit, name):
    optic_fits = fits.open(file_loc)
    optic_fits[0].data = np.float_(optic_fits[0].data) # typecasting
    if optic_type == 'opd':
        optic_surf = poppy.FITSOpticalElement(name = name, opd=optic_fits, opdunits = opdunit)
    else:
        optic_surf = poppy.FITSOpticalElement(name = name, transmission=optic_fits)
    return optic_surf

# Primary Mirror Surface
M1_surf = surfFITS(file_loc='data/ClayM1_0mask_meters_new.fits', optic_type='opd', opdunit='meters', 
                   name='M-1 surface')
# Secondary Mirror Surface
M2_surf = surfFITS(file_loc='data/M2_fitpsd.fits', optic_type='opd', opdunit='nanometers', 
                   name='M-2 surface')
# Tertiary Mirror Surface
M3_surf = surfFITS(file_loc='data/M3_fitpsd.fits', optic_type='opd', opdunit='nanometers', 
                   name='M-3 surface')

# OAPs
OAP0_surf = surfFITS(file_loc='data/oap_HP_0.fits', optic_type='opd', opdunit='nanometers', name='OAP-0 surface')

# Flats
peri1_surf = surfFITS(file_loc='data/flat_l100_0.fits', optic_type='opd', opdunit='nanometers', 
                          name='F-1 surface')
peri2_surf = surfFITS(file_loc='data/flat_l100_1.fits', optic_type='opd', opdunit='nanometers', 
                          name='F-2 surface')
K1_surf = surfFITS(file_loc='data/flat_l100_2.fits', optic_type='opd', opdunit='nanometers', 
                          name='K-1 surface')
K2_surf = surfFITS(file_loc='data/flat_l100_3.fits', optic_type='opd', opdunit='nanometers', 
                          name='K-2 surface')
K3_surf = surfFITS(file_loc='data/flat_l100_4.fits', optic_type='opd', opdunit='nanometers', 
                          name='K-3 surface')

pupil = surfFITS(file_loc='data/MagAOX_f11_pupil_256_unmasked.fits', optic_type='trans', opdunit='none', 
                 name='MagAO-X Pupil (unmasked)')

magaox = poppy.FresnelOpticalSystem(pupil_diameter=2*M1_radius, 
                                       npix=npix*2,
                                       beam_ratio=.34)
# Entrance Aperture
magaox.add_optic(poppy.CircularAperture(radius=M1_radius))

# Add Pupil
magaox.add_optic(pupil)

# Surface: Primary Mirror
magaox.add_optic(M1_surf)
magaox.add_optic(M1)
magaox.add_optic(poppy.CircularAperture(radius=M1_radius,name="M-1 aperture"))

# Surface: Secondary Mirror
magaox.add_optic(M2_surf, distance=d_m1_m2)
magaox.add_optic(M2)
magaox.add_optic(poppy.CircularAperture(radius=M2_radius,name="M-2 aperture"))

# Surface: Tertiary mirror 
magaox.add_optic(M3_surf, distance=d_m2_m3)
magaox.add_optic(poppy.CircularAperture(radius=M3_radius, name="M-3 aperture"))

# Surface: Periscope Mirror 1 (F-1)
magaox.add_optic(peri1_surf, distance=d_m3_wfsdichroic+d_wfsdichroic_peri1)
magaox.add_optic(poppy.CircularAperture(radius=flat_radius, name="F-1 aperture"))

# Surface: F/11 Focal Plane
magaox.add_optic(poppy.ScalarTransmission(planetype=PlaneType.intermediate, 
                                          name="F/11 focal plane (uncorrected)"), 
                                          distance=d_peri1_f11fp)

magaox.add_optic(poppy.ScalarTransmission(planetype=PlaneType.intermediate, 
                                          name="F/11 focal plane (corrected)"), 
                                          distance=f11_delta)

# Surface: Periscope Mirror 2 (F-2)
magaox.add_optic(peri2_surf, distance=d_f11fp_peri2)
magaox.add_optic(poppy.CircularAperture(radius=flat_radius, name="F-2 aperture"))

# Surface: OAP-0 (O-0)
magaox.add_optic(OAP0_surf, distance=d_peri2_oap0)
magaox.add_optic(OAP0)
magaox.add_optic(poppy.CircularAperture(radius=oap_radius,name="OAP-0 aperture"))

# Begin K-mirror setup
# Surface: K-1
magaox.add_optic(K1_surf, distance=d_oap0_k1)
magaox.add_optic(poppy.CircularAperture(radius=flat_radius, name="K-1 aperture"))

# Surface: K-2
magaox.add_optic(K2_surf, distance=d_k1_k2)
magaox.add_optic(poppy.CircularAperture(radius=flat_radius, name="K-2 aperture"))

# Surface: K-3
magaox.add_optic(K3_surf, distance=d_k2_k3)
magaox.add_optic(poppy.CircularAperture(radius=flat_radius, name="K-3 aperture"))

# Surface: woofer DM mirror
magaox.add_optic(poppy.ScalarTransmission(planetype=PlaneType.intermediate, 
                                          name="woofer DM"), 
                                          distance=d_k3_woofer)

plt.figure(figsize=[30,30])
woof_psf, woof_wfs = magaox.calcPSF(wavelength=testWavelength, display_intermediates=True, return_intermediates=True)

woof_num = len(woof_wfs)-1
woof_intensity = woof_wfs[woof_num].asFITS('intensity')

woof_pixscl = woof_intensity[0].header['PIXELSCL']
woof_pixscl

plt.figure(figsize=[8,8])
plt.imshow(woof_intensity[0].data, cmap='gray')
plt.colorbar()

woof_intensity.writeto('output/woof_intensity_zemaxRAG_zemaxM1M2.fits')

woof_diam = woof_pixscl * 512
woof_diam

woof_diam = woof_pixscl * 517.24
woof_diam



