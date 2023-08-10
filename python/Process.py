import lsst.afw.table
import lsst.afw.image
import lsst.afw.math
import lsst.meas.algorithms
import lsst.meas.base
import lsst.meas.deblender
import numpy as np
import astropy.io.fits as fits
import descwl
import astropy.table
import scipy.spatial
import scipy.ndimage
sims = False
if sims:
    from lsst.sims.GalSimInterface.wcsUtils import tanSipWcsFromDetector
    from lsst.sims.GalSimInterface import LSSTCameraWrapper
    from lsst.sims.utils import ObservationMetaData
schema = lsst.afw.table.SourceTable.makeMinimalSchema()
config1 = lsst.meas.algorithms.SourceDetectionConfig()
min_pix = 1
bkg_bin_size = 32
thr_value = 5
hsm = False
# Tweaks in the configuration that can improve detection
# Change carefully!
#####
config1.tempLocalBackground.binSize=bkg_bin_size # This changes the local background binning. The default is 32 pixels
config1.minPixels=min_pix # This changes the minimum size of a source. The default is 1
config1.thresholdValue=thr_value # This changes the detection threshold for the footprint (5 is the default)
#####
detect = lsst.meas.algorithms.SourceDetectionTask(schema=schema, config=config1)
deblend = lsst.meas.deblender.SourceDeblendTask(schema=schema)
config1 = lsst.meas.base.SingleFrameMeasurementConfig()
## HSM is not included in the stack by default. You have to download it and activate it.
if hsm:
    import lsst.meas.extensions.shapeHSM
    config1.plugins.names.add('ext_shapeHSM_HsmShapeBj')
    config1.plugins.names.add('ext_shapeHSM_HsmShapeLinear')
    config1.plugins.names.add('ext_shapeHSM_HsmShapeKsb')
    config1.plugins.names.add('ext_shapeHSM_HsmShapeRegauss')
    config1.plugins.names.add('ext_shapeHSM_HsmSourceMoments')
    config1.plugins.names.add('ext_shapeHSM_HsmPsfMoments')
if sims:
    camera_wrapper = LSSTCameraWrapper()
    obs = ObservationMetaData(pointingRA=0, pointingDec=0,
                        boundType='circle', boundLength=2.0,
                        mjd=52000.0, rotSkyPos=0,
                        bandpassName='i')
measure = lsst.meas.base.SingleFrameMeasurementTask(schema=schema, config=config1)

def process(input_path, output_path=None,seed=123):
    LSST_i = descwl.output.Reader(input_path).results # We read the image using descwl's package
    LSST_i.add_noise(noise_seed=seed) # We add noise
    image = lsst.afw.image.ImageF(LSST_i.survey.image.array) # We translate the image to be stack-readable
    sky_magnitude = LSST_i.survey.sky_brightness + LSST_i.survey.extinction*(LSST_i.survey.airmass -1.2)
    sky_counts = LSST_i.survey.exposure_time*LSST_i.survey.zero_point*10**(-0.4*(sky_magnitude-24))*LSST_i.survey.pixel_scale**2
    variance_array = LSST_i.survey.image.array+sky_counts # We generate a variance array
    variance = lsst.afw.image.ImageF(variance_array) # Generate the variance image
    masked_image = lsst.afw.image.MaskedImageF(image, None, variance) # Generate a masked image, i.e., an image+mask+variance image (with mask=None)
    psf_array = LSST_i.survey.psf_image.array # We read the PSF image from the package
    psf_array = psf_array.astype(np.float64) 
    psf_new = scipy.ndimage.zoom(psf_array,zoom=75/76.) # We have to rescale to have odd dimensions
    im = lsst.afw.image.ImageD(psf_new) # Convert to stack's format
    fkernel = lsst.afw.math.FixedKernel(im) 
    psf = lsst.meas.algorithms.KernelPsf(fkernel) # Create the kernel in the stack's format
    exposure = lsst.afw.image.ExposureF(masked_image) # Passing the image to the stack
    exposure.setPsf(psf) # Assign the exposure the PSF that we created
    if sims:
        wcs_in = tanSipWcsFromDetector('R:2,2 S:1,1',camera_wrapper,obs,2000) # We generate a WCS
        exposure.setWcs(wcs_in) # And assign it to the exposure
    table = lsst.afw.table.SourceTable.make(schema)  # this is really just a factory for records, not a table
    detect_result = detect.run(table, exposure) # We run the stack (the detection task)
    catalog = detect_result.sources   # this is the actual catalog, but most of it's still empty
    deblend.run(exposure, catalog) # run the deblending task
    measure.run(catalog, exposure) # run the measuring task
    catalog = catalog.copy(deep=True)
    if output_path is not None:
        catalog.writeFits(output_path) #write a copy of the catalog
    return catalog # We return a catalog object

import os
btf_dir = '/global/projecta/projectdirs/lsst/groups/WL/projects/wl-btf/'

catalog = process(os.path.join(btf_dir,'LSST_i_lite.fits'), output_path='LSST_i_DM.fits.gz')

catalog.schema

get_ipython().system('ls -lrh /global/projecta/projectdirs/lsst/groups/WL/projects/wl-btf/*.fits*')

tab = catalog.asAstropy() #We can also convert the catalog to an astropy table and show the contents
tab



