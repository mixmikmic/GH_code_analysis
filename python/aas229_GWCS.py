from asdf import AsdfFile
import numpy as np
from astropy.modeling import models

# Create a 2D rotation model
rotation = models.Rotation2D(angle=60)
print(rotation)

# Open an ASDF file object
f = AsdfFile()

# Every ASDF file object has an attribute, called "tree"
# It is a dict like object which store theinformation in YAML format
print(f.tree)

f.tree['model'] = rotation
f.write_to('rotation.asdf')
#!less rotation.asdf

import numpy as np
from astropy.modeling import models
from astropy import units as u
from astropy import coordinates as coord
from asdf import AsdfFile
from gwcs import wcs
from gwcs import coordinate_frames as cf
from gwcs import wcstools
from gwcs import utils as gwutils

polyx = models.Polynomial2D(4)
polyx.parameters = np.random.randn(15)
polyy = models.Polynomial2D(4)
polyy.parameters = np.random.randn(15)
distortion = (models.Mapping((0, 1, 0, 1)) | polyx & polyy).rename("distortion")

f = AsdfFile()
f.tree['model'] = distortion
f.write_to('poly.asdf', all_array_storage='inline')
#!less poly.asdf

undist2sky = (models.Shift(-10.5) & models.Shift(-13.2) | models.Rotation2D(0.0023) |               models.Scale(.01) & models.Scale(.04) | models.Pix2Sky_TAN() |               models.RotateNative2Celestial(5.6, -72.05, 180)).rename("undistorted2sky")

detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"), unit=(u.pix, u.pix))
sky_frame = cf.CelestialFrame(name="icrs", reference_frame=coord.ICRS())
focal_frame = cf.Frame2D(name="focal_frame", unit=(u.arcsec, u.arcsec))

pipeline = [(detector_frame, distortion),
            (focal_frame, undist2sky),
            (sky_frame, None)
            ]
wcsobj = wcs.WCS(pipeline)
print(wcsobj)

# Calling the WCS object like a function evaluates the transforms.
ra, dec = wcsobj(500, 600)
print(ra, dec)

# Display the frames available in the WCS pipeline
print(wcsobj.available_frames)

wcsobj.input_frame

wcsobj.output_frame

# Because the output_frame is a CoordinateFrame object we can get as output
# coordinates.SkyCoord objects.
skycoord = wcsobj(1, 2, output="numericals_plus")
print(skycoord)

print(skycoord.transform_to('galactic'))

print(wcsobj.output_frame.coordinates(ra, dec))

# It is possible to retrieve the transform between any
# two coordinate frames in the WCS pipeline
print(wcsobj.available_frames)

det2focal = wcsobj.get_transform("detector", "focal_frame")
fx, fy = det2focal(1, 2)
print(fx, fy)

# And we can see what the units are in focal_frame
print(wcsobj.focal_frame.coordinates(fx, fy))

# It is also possible to replace a transform 
# Create a transforms which shifts in X and y
new_det2focal = models.Shift(3) & models.Shift(12)
# Replace the transform between "detector" and "v2v3"
wcsobj.set_transform("detector", "focal_frame", new_det2focal)
new_ra, new_dec = wcsobj(500, 600)
print(ra, dec)
print(new_ra, new_dec)

# We can insert a transform in the pipeline just before or after a frame
rotation = models.EulerAngleRotation(.1, 12, 180, axes_order="xyz")
wcsobj.insert_transform("focal_frame", rotation)
wcsobj.get_transform("detector", "focal_frame")(1, 2)

from jwst import datamodels
nrs_fs = "nrs1_assign_wcs.fits.gz"
nrs = datamodels.ImageModel(nrs_fs)
from jwst.assign_wcs import nirspec
slits = nirspec.get_open_slits(nrs)
print(slits[0])

slits = nirspec.get_open_slits(nrs)
for s in slits:
    print(s)

s0 = nirspec.nrs_wcs_set_input(nrs, "S200A1")
print(s0.domain)

s0.available_frames

s0.output_frame

x, y = wcstools.grid_from_domain(s0.domain)

ra, dec, lam = s0(x, y)

res = s0(1000, 200, output="numericals_plus")
print(res)

get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
plt.imshow(lam, aspect='auto')
plt.title("lambda, microns")
plt.colorbar()



