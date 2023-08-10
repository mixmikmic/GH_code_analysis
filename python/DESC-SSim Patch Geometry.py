import sys
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.daf.persistence as dafPersist

import seaborn as sns
sns.set(context='talk',
        style='whitegrid',
        palette='deep',
        font='sans-serif',
        font_scale=0.8,
        color_codes=True,
        rc={'text.usetex': False})

get_ipython().magic('matplotlib inline')

DATA_DIR = '.'
# DATA_DIR = '/global/cscratch1/sd/descdm/DC1'  # ON NERSC

# To access the skymap construct a butler with the repo and ask for the "deepCoadd_skyMap"
butler = dafPersist.Butler(os.path.join(DATA_DIR, 'full_focalplane_undithered'))
skyMap = butler.get("deepCoadd_skyMap")

def makePatch(vertexList, wcs):
    """Return a path in sky coords from vertex list in pixel coords"""
    
    skyPatchList = [wcs.pixelToSky(pos).getPosition(afwGeom.degrees) for pos in vertexList]
    verts = [(coord[0], coord[1]) for coord in skyPatchList]
    verts.append((0,0))
    codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY,
             ]
    return Path(verts, codes)   

def plotSkyMap(skyMap, tract=0, title="Patch Geometry"):
   tractInfo = skyMap[tract]
   tractBox = afwGeom.Box2D(tractInfo.getBBox())
   tractPosList = tractBox.getCorners()
   wcs = tractInfo.getWcs()
   xNum, yNum = tractInfo.getNumPatches()

   fig = plt.figure(figsize=(12,8))
   ax = fig.add_subplot(111)
   for x in range(xNum):
       for y in range(yNum):
           patchInfo = tractInfo.getPatchInfo([x, y])
           patchBox = afwGeom.Box2D(patchInfo.getOuterBBox())
           pixelPatchList = patchBox.getCorners()
           path = makePatch(pixelPatchList, wcs)
           patch = patches.PathPatch(path, alpha=0.1, lw=1)
           ax.add_patch(patch)
           center = wcs.pixelToSky(patchBox.getCenter()).getPosition(afwGeom.degrees)
           ax.text(center[0], center[1], '%d,%d'%(x,y), size=6, ha="center", va="center")

   skyPosList = [wcs.pixelToSky(pos).getPosition(afwGeom.degrees) for pos in tractPosList]
   ax.set_xlim(max(coord[0] for coord in skyPosList) + 1,
               min(coord[0] for coord in skyPosList) - 1)
   ax.set_ylim(min(coord[1] for coord in skyPosList) - 1, 
               max(coord[1] for coord in skyPosList) + 1)

   ax.set_xlabel("RA (deg.)")
   ax.set_ylabel("Dec (deg.)")
   ax.set_title(title)
   return ax

for directory in ['full_focalplane_undithered',
                 'DC1-imsim-dithered',
                 'DC1-phoSim-3a']:
    butler = dafPersist.Butler(os.path.join(DATA_DIR, directory))
    skyMap = butler.get("deepCoadd_skyMap")
    plotSkyMap(skyMap, tract=0, title=directory)

tractInfo = skyMap[0]
wcs = tractInfo.getWcs()

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
plotBox = afwGeom.Box2D(tractInfo.getPatchInfo([0, 0]).getOuterBBox())
for x in range(2):
    for y in range(2):
        # Plot outer BBOX
        patchInfo = tractInfo.getPatchInfo([x, y])
        patchBox = afwGeom.Box2D(patchInfo.getOuterBBox())
        plotBox.include(patchBox)
        pixelPatchList = patchBox.getCorners()
        path = makePatch(pixelPatchList, wcs)
        patch = patches.PathPatch(path, alpha=0.2, lw=1)
        ax.add_patch(patch)
        # Plot inner BBox
        patchBox = afwGeom.Box2D(patchInfo.getInnerBBox())
        pixelPatchList = patchBox.getCorners()
        path = makePatch(pixelPatchList, wcs)
        patch = patches.PathPatch(path, fill=None, lw=1, 
                                  linestyle='dotted', color='k')
        ax.add_patch(patch)
               
        center = wcs.pixelToSky(patchBox.getCenter()).getPosition(afwGeom.degrees)
        ax.text(center[0], center[1], 'patchID=%d,%d'%(x,y), size=12, ha="center", va="center")

skyPosList = [wcs.pixelToSky(pos).getPosition(afwGeom.degrees) for pos in plotBox.getCorners()]
ax.set_xlim(max(coord[0] for coord in skyPosList),
            min(coord[0] for coord in skyPosList))
ax.set_ylim(min(coord[1] for coord in skyPosList), 
            max(coord[1] for coord in skyPosList))
ax.set_xlabel("RA (deg.)")
ax.set_ylabel("Dec (deg.)")
ax.set_title("Zoom in")

