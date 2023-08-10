import collections
import numpy as np

import lsst.daf.persistence as dafPersist
import lsst.afw.image as afwImage
import lsst.afw.detection as afwDetect
import lsst.afw.geom as afwGeom
import lsst.afw.cameraGeom.utils as cameraGeomUtils
import lsst.afw.display.ds9 as ds9
import lsst.afw.display.utils as displayUtils
import lsst.afw.display.rgb as afwRgb

import os
os.environ["SUPRIME_DATA_DIR"] = "/work/yasuda/rerun/yasuda/deep2"
refDates = {"HSC-G": "2014-10-01",
            "HSC-R": "2014-09-22",
            "HSC-I": "2014-09-22",
            "HSC-Z": "2014-10-01",
            "HSC-Y": "2014-09-18"}
imgDates = {'HSC-G': '2015-07-14',
            'HSC-R': '2015-07-15',
            'HSC-I': '2015-07-20',
            'HSC-Z': '2015-07-22',
            'HSC-Y': '2015-07-23'}

butlers = {}
for f in imgDates:
    rerun = imgDates[f]
    dataPath = os.path.join(os.environ["SUPRIME_DATA_DIR"], "rerun", rerun)

    butlers[f] = dafPersist.Butler(dataPath)
skymap = butlers.values()[0].get("deepCoadd_skyMap")

# butler type for difference images
if False:
    deepCoadd_diff = "deepCoadd_tempExp_diff"
    deepCoadd_direct = "deepCoadd_tempExp"  
else:
    deepCoadd_diff = "deepCoadd_diff"
    deepCoadd_direct = "deepCoadd_calexp"

def getButler(dataId):
    return butlers[dataId["filter"]]

def assembleTiles(images):
    """Assemble a list of tiles according to their XY0 values"""
    bigBBox = afwGeom.BoxI()

    for im in images:
        bigBBox.include(im.getBBox(afwImage.PARENT))

    bigIm = afwImage.MaskedImageF(bigBBox)
    for im in images:
        if True:
            sub = bigIm.Factory(bigIm, im.getBBox(afwImage.PARENT), afwImage.PARENT)
            sub <<= im.getMaskedImage()
            del sub
        else:
            bigIm[im.getBBox(afwImage.PARENT)] = im

    return bigIm

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def assemblePatches(patches, patchInfo):
    """Assemble a list of tiles according to their XY0 values"""
    bigBBox = afwGeom.BoxI()
    
    for p in patches:
        pi = patchInfo[p]
        bigBBox.include(pi.getInnerBBox())

    bigIm = afwImage.MaskedImageF(bigBBox)
    for p in patches:
        im = patches[p].getMaskedImage()
        pi = patchInfo[p]

        if True:
            sub = bigIm.Factory(bigIm, pi.getInnerBBox(), afwImage.PARENT)
            sub <<= im.Factory(im, pi.getInnerBBox(), afwImage.PARENT)
            del sub
        else:
            bigIm[pi.getInnerBBox()] = im[pi.getInnerBBox()]

    wcs = patches[p].getWcs().clone()
    wcs.shiftReferencePixel(afwGeom.ExtentD(bigIm.getXY0() - patches[p].getXY0()))

    return afwImage.makeExposure(bigIm, wcs)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def getTract(butler, dataId, pps=None, butlerType="deepCoadd_calexp"):
    skymap = butler.get("deepCoadd_skyMap")
    patchInfo = dict(('%d,%d' % _.getIndex(), _) for _ in skymap[dataId["tract"]])

    if pps is None:
        pps = list(patchInfo)
        
    filterName = dataId["filter"]
    if len(filterName) == 1:
        filterName = "HSC-%s" % filterName.upper()
        
    patches = {}
    for pp in pps:
        patches[pp] = butler.get(butlerType, dataId, filter=filterName, patch=pp)

    return assemblePatches(patches, patchInfo)

def getPatchInfo(butler, dataId, pps=None, butlerType="deepCoadd_calexp"):
    skymap = butler.get("deepCoadd_skyMap")
    patchInfo = dict(('%d,%d' % _.getIndex(), _) for _ in skymap[dataId["tract"]])
    
    if pps is None:
        pps = list(patchInfo)

    filterName = dataId["filter"]
    if len(filterName) == 1:
        filterName = "HSC-%s" % filterName.upper()
        
    info = collections.OrderedDict()
    for pp in pps:
        patch = butler.get(butlerType, dataId, filter=filterName, patch=pp)
        info[pp] = patch.getInfo()

    return info

dataId = dict(tract=9463, filter='HSC-G', patch='3,6')
if False:
    p = getButler(dataId).get(deepCoadd_direct, dataId, immediate=True)
    ds9.mtv(p, frame=10)

if True:
    info = getPatchInfo(getButler(dataId), dataId=dataId, pps=['3,6'])

for pp in info:
    print pp
    inputs = info[pp].getCoaddInputs()
    w = 0
    for v in inputs.visits:
        w += v["weight"]
    for v in inputs.visits:
        print "\t%5d  %.3f" % (v.getId(), v["weight"]/w)

if False:
    d = getButler(dataId).get(deepCoadd_diff, dataId, visit=24164, immediate=True)
    ds9.mtv(d, frame=0)

pps = ["%d,%d" % (i , j) for i in range(3, 6) for j in range(3, 6)]
#pps = ["%d,%d" % (i , j) for i in range(5, 6) for j in range(5, 6)]
pps = ['2,6', '3,6']
visits = [-1]   # we're subtracting coadds from each other
#visits = [24164, 24196, 24228]

fullPatchIm = getTract(getButler(dataId), dataId=dataId, pps=pps,
                       butlerType="deepCoadd_calexp")
patchIm = fullPatchIm

if False:
    ds9.mtv(patchIm, frame=1, title="%d %s" % (dataId["tract"], str(pps)[1:-1].replace("'", "")))

dataId.update(filter='HSC-R')
diffIms = collections.OrderedDict()
for frame, v in enumerate(visits, 2):
    if v in diffIms:
        continue

    diffIms[v] = getTract(getButler(dataId), dataId=dict(visit=v, **dataId), pps=pps,
                          butlerType=deepCoadd_diff)
    if False:
        ds9.mtv(diffIms[v], frame=frame, title=v if v > 0 else dataId["filter"])

visitIms = collections.OrderedDict()
for frame, v in enumerate(diffIms.keys(), 2 + len(visits)):
    if v in visitIms:
        continue

    visitIms[v] = getTract(getButler(dataId), dataId=dict(visit=v, **dataId), pps=pps,
                           butlerType=deepCoadd_direct)
    if False:
        ds9.mtv(visitIms[v], frame=frame, title=v if v > 0 else dataId["filter"])

diffImsC = collections.OrderedDict()
visitImsC = collections.OrderedDict()

filters = ['HSC-G', 'HSC-R', 'HSC-I']
for frame, f in enumerate(filters, 2):
    dataId['filter'] = f

    if f not in diffImsC:
        diffImsC[f] = getTract(getButler(dataId), dataId=dict(**dataId), pps=pps,
                              butlerType=deepCoadd_diff)
    if True:
        ds9.mtv(diffImsC[f], frame=frame, title=dataId["filter"])

    if f not in visitImsC:
        visitImsC[f] = getTract(getButler(dataId), dataId=dict(**dataId), pps=pps,
                               butlerType=deepCoadd_direct)
    if True:
        ds9.mtv(visitImsC[f], frame=frame + len(filters), title=dataId["filter"])

def maskArtefacts(patchIm, diffIms, threshold=2, nGrow=15, brightThreshold=np.inf,
                  clippedNpixMin=1000, includeChipGaps=True):
    """
    threshold : nSigma (but the data's correlated, so this isn't directly interpretable)
    nGrow     : pixels; saturated footprints only
    """
    fs = afwDetect.FootprintSet(patchIm.getMaskedImage(), 
                                afwDetect.Threshold(threshold, afwDetect.Threshold.PIXEL_STDEV))

    x0, y0 = patchIm.getXY0()
    mask = patchIm.getMaskedImage().getMask()
    image = patchIm.getMaskedImage().getImage()

    CLIPPED = 1 << mask.getMaskPlane("CLIPPED")
    SAT = 1 << mask.getMaskPlane("SAT")
    INTRP = 1 << mask.getMaskPlane("INTRP")

    maskedPixels = (mask.getArray() & SAT)
    height, width = maskedPixels.shape

    for foot in fs.getFootprints():
        peak = foot.getPeaks()[0]
        x, y, = peak.getIx(), peak.getIy()
        setBits = foot.overlapsMask(mask)
        if image.get(x - x0, y - y0) > brightThreshold or (setBits & SAT) or            ((setBits & CLIPPED) and foot.getNpix() >= clippedNpixMin):
            if (setBits & SAT) and nGrow:
                foot = afwDetect.growFootprint(foot, nGrow)
            for s in foot.getSpans():
                sy = s.getY() - y0
                sx0, sx1 = s.getX0() - x0, s.getX1() - x0

                try:
                    maskedPixels[sy, sx0:sx1+1] = 1
                except IndexError:
                    if sy < 0 or sy >= height:
                        continue
                    if sx0 < 0:
                        sx0 = 0
                    if sx1 >= width:
                        sx1 = width - 1
                    maskedPixels[sy, sx0:sx1+1] = 1

    if includeChipGaps:             # include union of chip gaps in maskedPixels
        for v in diffIms:
            maskedPixels[np.isnan(diffIms[v][bbox].getMaskedImage().getImage().getArray())] = 1

    cleaned = collections.OrderedDict()
    for v in diffIms:
        cleaned[v] = diffIms[v][bbox].clone()
        imArr = cleaned[v].getMaskedImage().getImage().getArray()
        mskArr = cleaned[v].getMaskedImage().getMask().getArray()
        mskArr &= CLIPPED

        bad = np.logical_or(np.isnan(imArr), maskedPixels > 0)        
        good = np.logical_and(np.isfinite(imArr), maskedPixels == 0)

        q25, q50, q75 = np.percentile(imArr[good], [25, 50, 75])
        imArr[bad] = np.random.normal(q50, 0.741*(q75 - q25), size=np.sum(bad))
        mskArr[bad] |= INTRP

    return cleaned

if True:
    bbox = fullPatchIm.getBBox()
else:
    bbox = afwGeom.BoxI(afwGeom.PointI(9000, 9000), afwGeom.PointI(12000 - 1, 12000 - 1))

patchIm = fullPatchIm[bbox]

if False:
    ds9.mtv(patchIm, frame=1, title="%d %s" % (tract, str(pps)[1:-1].replace("'", "")))

cleaned = maskArtefacts(patchIm, diffIms if False else diffImsC)

if False:
    frame0 = 2
    for frame, v in enumerate(diffIms, frame0):
        title = v if len(diffIms) > 12000 else dataId["filter"]

        if True:
            ds9.mtv(cleaned[v], frame=frame, title=title)
        if True:
            ds9.mtv(visitImsC[v][bbox], frame=frame + len(diffIms), title=title)

if len(cleaned) == 1:
    R, G, B = [im.getMaskedImage().getImage().clone() for im in 3*cleaned.values()]
    G *= 0
    B *= 0
else:
    R, G, B = reversed([im.getMaskedImage().getImage().clone() for im in cleaned.values()])

    # make the blue a bit brighter by mixing in some green
    if isinstance(cleaned.keys()[0], int):
        G.getArray()[:] += 0.5*B.getArray()

# Set the pixel masks
mskVal = 0.025
for im in [G, B]:
    im.getArray()[bad] += mskVal
    
clippedPixels = (patchIm.getMaskedImage().getMask().getArray() & CLIPPED) == CLIPPED
for im in [R, B]:
    im.getArray()[clippedPixels] += mskVal

#
# Add in the ghost of the direct image
#
addGhost = False
if addGhost:
    for im in [R, G, B]:
        im.getArray()[:] += 0.2*patchIm.getMaskedImage().getImage().getArray()
    
rgb = afwRgb.makeRGB(R, G, B, 
                     minimum=-0.1, range=1, Q=4
                     #minimum=-.2, range=.4, Q=20
                    )
afwRgb.writeRGB("tract-%d-patches-diff%s.png" % (dataId["tract"], "-ghost" if addGhost else ""), rgb)

if False:
    afwRgb.displayRGB(rgb)

rgb = afwRgb.makeRGB(*reversed([im.getMaskedImage().getImage() for im in visitImsC.values()]), 
                     minimum=-0.1, range=1, Q=4)
afwRgb.writeRGB("tract-%d-patches-%s.png" % (dataId["tract"], "gri"), rgb)

gray = afwRgb.makeRGB(*3*[patchIm[bbox].getMaskedImage()], minimum=0, range=4, Q=8)
afwRgb.writeRGB("tract-%d-patches-%s.png" % (dataId["tract"], "gray"), gray)

if False:
    ds9.mtv(butler.get("deepCoadd_tempExp", tract=0, patch='3,3', filter='HSC-G', visit=24228))

import lsst.afw.cameraGeom as afwCamGeom
boreSight = calexp.getDetector().getCenter()   # pixels in FPA coordinates
pointing = calexp.getWcs().pixelToSky(boreSight.getPixels(1)) # where the boresight's pointing
print [_.asDegrees() for _ in pointing]

