# Import Python libraries to work with SciServer
import SciServer.CasJobs as CasJobs # query with CasJobs
import SciServer.SciDrive as SciDrive   # read/write to/from SciDrive
import SciServer.SkyServer as SkyServer   # show individual objects and generate thumbnail images through SkyServer
print('SciServer libraries imported')

# Import other libraries for use in this notebook.
import numpy as np                  # standard Python lib for math ops
from scipy.misc import imsave       # save images as files
import pandas                       # data manipulation package
import matplotlib.pyplot as plt     # another graphing package
import os                           # manage local files in your Compute containers
print('Supporting libraries imported')

# Apply some special settings to the imported libraries
# ensure columns get written completely in notebook
pandas.set_option('display.max_colwidth', -1)
# do *not* show python warnings 
import warnings
warnings.filterwarnings('ignore')
print('Settings applied')

# Find objects in the Sloan Digital Sky Survey's Data Release 14.
#
# Query the Sloan Digital Sky Serveys' Data Release 14.
# For the database schema and documentation see http://skyserver.sdss.org/dr14
#
# This query finds "a 4x4 grid of nice-looking galaxies": 
#   galaxies in the SDSS database that have a spectrum 
#   and have a size (petror90_r) larger than 10 arcsec.
# 
# First, store the query in an object called "query"
query="""
SELECT TOP 16 p.objId,p.ra,p.dec,p.petror90_r, p.g, p.r
  FROM galaxy AS p
   JOIN SpecObj AS s ON s.bestobjid = p.objid
WHERE p.u BETWEEN 0 AND 19.6
  AND p.g BETWEEN 0 AND 17  AND p.petror90_r > 10
"""
gals = CasJobs.executeQuery(query, "dr14")
gals = gals.set_index('objId')
gals

# Find objects in the Sloan Digital Sky Survey's Data Release 14.
# First, store the query in an object called "query"
query="""
SELECT TOP 16 p.objId,p.ra,p.dec,p.petror90_r, p.g, p.r
  FROM galaxy AS p
   JOIN SpecObj AS s ON s.bestobjid = p.objid
WHERE p.u BETWEEN 0 AND 19.6
  AND p.g BETWEEN 0 AND 17
  AND p.petror90_r > 10
"""
gals = CasJobs.executeQuery(query, "dr14")
gals = gals.set_index('objId')
gals

# Try making an Ra, Dec scatter plot...

plt.scatter(gals['ra'], gals['dec'])
plt.show() 

# make scatterplot
#plt.scatter(gals['g'],gals['r'])

# create an overall figure object (fig) and an "axis" object to display this data (ax)
fig, ax = plt.subplots()

# This graph is associated with the axis object
ax.scatter(gals['ra'], gals['dec'])

# Add formatting options here. Some may apply to fig and others to ax.

# Show the graph will all the options you set.
plt.show()

# store result as HDF5 file 
h5store = pandas.HDFStore('GalaxyThumbSample.h5')
h5store['galaxies']=gals
h5store.close()

# store result as CSV file
gals.to_csv('GalaxyThumbSample.csv')

print ("Done.")

# store result as HDF5 file 
h5store = pandas.HDFStore('GalaxyThumbSample.h5')
h5store['galaxies']=gals
h5store.close()

# store result as CSV file
gals.to_csv('GalaxyThumbSample.csv')

print ("Done.")

# set thumbnail parameters
width=200           # image width
height=200          # height
pixelsize=0.396     # image scale
plt.figure(figsize=(15, 15))   # display in a 4x4 grid
subPlotNum = 1

i = 0
nGalaxies = len(gals)
for index,gal in gals.iterrows():           # iterate through rows in the DataFrame
    i = i + 1
    print('Getting image '+str(i)+' of '+str(nGalaxies)+'...')
    if (i == nGalaxies):
        print('Plotting images...')
    scale=2*gal['petror90_r']/pixelsize/width
    img = SkyServer.getJpegImgCutout(ra=gal['ra'], dec=gal['dec'], width=width, height=height, scale=scale,dataRelease='DR14')
    plt.subplot(4,4,subPlotNum)
    subPlotNum += 1
    plt.imshow(img)                               # show images in grid
    plt.title(index)                            # show the object identifier (objId) above the image.

# Step 7b: Specify the directory in your SciDrive to hold the thumbnail images
mydir = 'week5'

# if that directory already exists, delete it before trying to create it again.
directories_dict = SciDrive.directoryList()
for x in directories_dict['contents']:
    if (x['path'][1:] == mydir):
        SciDrive.delete(x['path'])     

# create that directory in your SciDrive
SciDrive.createContainer(mydir)
print('Created SciDrive directory: ',mydir,'\n')


# set thumbnail parameters
width=200           # image width
height=200          # height
pixelsize=0.396     # image scale

# make a name for your "manifest" -- the table that matches your images to known information about the galaxies
fout = open("galaxy_manifest.csv", 'w')
fout.write("index, image, ra, dec \n")

# Write thumbnails to SciDrive. You will see a confirmation message when each is written
i = 0
puburls=[]
for index,gal in gals.iterrows():   
    i = i + 1
    print('Writing image file '+str(i)+' of '+str(len(gals))+'...')
    scale=2*gal['petror90_r']/pixelsize/width
    img = SkyServer.getJpegImgCutout(ra=gal['ra'], dec=gal['dec'], width=width, height=height, scale=scale,dataRelease='DR14')
    localname = str(index)+'.jpg'
    scidrivename = mydir+"/"+localname
# Here the file gets uploaded to the container 
    imsave(localname,img)
    SciDrive.upload(scidrivename, localFilePath=localname)
    puburls.append(SciDrive.publicUrl(scidrivename))
    os.remove(localname)
    fout.write('{0:19.0f}, {1:.0f}.jpg, {2:.6f}, {3:.6f}\n'.format(index,index,gal['ra'],gal['dec']))
fout.close()

gals['pubURL']=puburls

print('\nWriting manifest...\n')
SciDrive.upload(mydir+'/galaxy_manifest.csv', localFilePath = 'galaxy_manifest.csv')

os.remove('galaxy_manifest.csv')
print('Done!')



