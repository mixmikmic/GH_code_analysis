import astropy.visualization as viz
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import astropy.wcs as WCS # you need this astropy module for processing fits files
import matplotlib as mpl
import numpy as np

# reading the header data unit (HDU) of the fits file
hdu = fits.open("xorastro_metadata.fits")
#-->returns an object called an hdu which is a list-like collection of HDU objects

hdu.info() #summarizes the content of the opened FITS file

# reading the fits header 
hdr = hdu[0].header
# reading the image data
img = hdu[1].data


img

print(np.shape(img))

RA = img["ra"] #right ascention
Dec = img["dec"] #declination

print(RA.shape)

# Plotting the RA and Dec of SDSS data
plt.figure(1, figsize=(20,10))
ax = plt.subplot(1,1,1)
ax.plot(RA, Dec, "r,", )
ax.set_title("Full size Galaxy Frame")
ax.set_xlabel("Right Ascension")
ax.set_ylabel("Declinatio")
plt.savefig("RA_Dec.png")
plt.show()

u = img["u"] 
g = img["g"]
i = img["i"]
r = img["r"]
z = img["z"]

plt.figure(1, figsize=(20,10))
ax = plt.subplot(1,1,1)
ax.plot(g-r, r-i, "ko")
ax.set_title("Color-Color diagram")
ax.set_xlabel("g-r")
ax.set_ylabel("r-i")
ax.set_ylim(-25, 25)
ax.set_xlim(-25, 25)
#plt.colorbar()
plt.savefig("color_color.png")
plt.show()



