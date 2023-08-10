## plot within the notebook
get_ipython().magic('matplotlib inline')
import warnings
## No annoying warnings
warnings.filterwarnings('ignore')
# - Astrobject Modules
from astrobject.utils.tools import load_pkl
from astrobject import get_target
from astrobject import get_image, get_instrument

# Load an image object by giving the fits file.
sdssg = get_image("data/sdss_PTF10qjq_g.fits")
# - Let's see how it looks like
pl = sdssg.show(logscale=True)

print pl

# The Rawdata (shown in logscale, which is the default of the show() method), using vmin and vmax as percentile values
pl = sdssg.show("rawdata",vmax="99",vmin="5")

# The background ; see the background mask array using backgroundmask
pl = sdssg.show("background", logscale=False)

pl = sdssg.show_background(vmin="1",vmax="99")

# -------------------------
# - Build the astrotarget
# -------------------------
# Basic data
dicosn = load_pkl("data/PTF10qjq_sninfo.pkl")
print dicosn

# Create the astrotarget
sn = get_target(name=dicosn["object"], zcmb=dicosn["zcmb"],
                 ra=dicosn["Ra"],dec=dicosn["Dec"],
                 type_=dicosn["type"],forced_mwebmv=dicosn["MWebmv"])

print sn.zcmb, sn.ra, sn.dec, sn.distmeter

# let's add a target
sdssg.set_target(sn)

# for the 1000,800 point:
print "pixel [1000,800] the corresponding Ra,Dec are", sdssg.pixel_to_coords(1000,800)
# for the target
print "pixel set target the corresponding pixels coords are",sdssg.coords_to_pixel(sn.ra,sn.dec)

pl = sdssg.show()

pl = sdssg.show(zoomon="target")

# Which also work with the background, rawdata etc.
pl = sdssg.show(toshow="background", zoomon='target', zoom=400, logscale=False)

sdssg.sep_extract()

pl = sdssg.show(show_sepobjects=True)

sdssg.sepobjects.data

sdssg.download_catalogue("sdss")

sdssg.show( show_catalogue=True, show_sepobjects=True)



