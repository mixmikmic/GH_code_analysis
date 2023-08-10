import numpy as np

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

import lsst.pex.exceptions as pexExcept
import lsst.daf.persistence as dafPersist
import lsst.afw.image as afwImage
import lsst.afw.detection as afwDetect
import lsst.afw.display.utils as afwDisplayUtils
import lsst.afw.display as afwDisp

butler = dafPersist.Butler(os.path.join("/Users/rhl/PFS/Data/rerun", "rhl", "tmp"))
#afwDisplay.setDefaultBackend("lsst.display.ginga")
afwDisplay.Display.delAllDisplays()
disp = afwDisplay.Display(1, open=True)
disp2 = afwDisplay.Display(2, open=True)

figDir = os.path.expanduser("~/TeX/Talks/PFS/Princeton-2016-09") if False else None

dataId = dict(arm="r", spectrograph=1)

calexp = butler.get("calexp", dataId, visit=7304)

calexp.setWcs(None)
disp.mtv(calexp)

mi = calexp.getMaskedImage()
mi -= np.median(mi.getImage().getArray())
fp = afwDetect.FootprintSet(mi, afwDetect.Threshold(50))

feet = afwDetect.FootprintList()
for foot in fp.getFootprints():
    if foot.getNpix() < 10 and len(foot.getPeaks()) == 1:
        feet.append(foot)
fp = afwDetect.FootprintSet(mi.getBBox())
fp.setFootprints(feet)

rGrow, isotropic = 7, True
fp = afwDetect.FootprintSet(fp, rGrow, isotropic)

feet = afwDetect.FootprintList()
for foot in fp.getFootprints():
    if len(foot.getPeaks()) == 1:
        feet.append(foot)
fe55 = afwDetect.FootprintSet(mi.getBBox())
fe55.setFootprints(feet)

msk = mi.getMask()
mi.getMask(); msk.clearMaskPlane(msk.getMaskPlane("DETECTED"))
fe55.setMask(mi.getMask(), "DETECTED")

if True:
    disp.mtv(mi)
    disp.scale('linear', 'zscale')    
else:
    pass
    disp.erase()
    
if False:
    with disp.Buffering():
        for foot in fe55.getFootprints():  
            afwDisplayUtils.drawFootprint(foot, display=disp, peaks=True,
                                          ctype=afwDisplay.GREEN)      

hsize = 15
stamp = afwImage.ImageF(2*hsize + 1, 2*hsize + 1); stamp[:] = 0
bkgd = []
nFe55 = 0
for foot in fe55.getFootprints():
    peak = foot.getPeaks()[0]
    x, y = peak.getI()
    try:
        fe55stamp = mi.getImage()[x - hsize:x + hsize + 1, y - hsize:y + hsize + 1]
        stamp += fe55stamp
        nFe55 += 1
        bkgd.append(np.median(fe55stamp.getArray()[0, :]))
    except pexExcept.LengthError as e:
        pass
    
stamp /= np.max(stamp.getArray()) if False else nFe55
disp2.mtv(stamp)

I00 = float(stamp[hsize, hsize])
I10 = np.array(
    [float(stamp[hsize + dx, hsize + dy]) for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]])
I11 = np.array(
    [float(stamp[hsize + dx, hsize + dy]) for dx, dy in [(1, 1), (1, -1), (1, 1), (-1, 1)]])

print("Fractional errors: I10 %.2g  I11 %.2g" % (np.std(I10/I00)/3, np.std(I11/I00)/3))

pixelSize = 15    # microns
print("n(Fe55) = %d, sigma = %.2f (%.2f) microns" % 
      (nFe55, 
       pixelSize*np.sqrt(1/(2*np.log(I00/np.mean(I10))) - 2*1/12.0),
       pixelSize*np.sqrt(2/(2*np.log(I00/np.mean(I11))) - 2*1/12.0),
                                                 ))

import lsst.afw.table as afwTable
import lsst.meas.base as measBase

centroidName = "base_GaussianCentroid"
shapeName = "base_SdssShape"

schema = afwTable.SourceTable.makeMinimalSchema()
schema.getAliasMap().set("slot_Centroid", centroidName)
schema.getAliasMap().set("slot_Shape", shapeName)

control = measBase.GaussianCentroidControl()
centroider = measBase.GaussianCentroidAlgorithm(control, centroidName, schema)

sdssShape = measBase.SdssShapeControl()
shaper = measBase.SdssShapeAlgorithm(sdssShape, shapeName, schema)
table = afwTable.SourceTable.make(schema)

exp = afwImage.makeExposure(afwImage.makeMaskedImage(stamp))
centerX, centerY = hsize, hsize
src = table.makeRecord()
foot = afwDetect.Footprint(exp.getBBox())
foot.addPeak(centerX, centerY, 1)
src.setFootprint(foot)

centroider.measure(src, exp)
shaper.measure(src, exp)

disp2.dot(src.getShape(), *src.getCentroid(), ctype=afwDisplay.BLUE)
shape = src.getShape()
print("n(Fe55) = %d, sigma (adaptive) = %.2f microns" % 
      (nFe55,  pixelSize*np.sqrt(0.5*(shape.getIxx() + shape.getIyy()))))

