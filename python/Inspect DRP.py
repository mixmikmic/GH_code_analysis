import numpy as np

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

import lsst.daf.persistence as dafPersist
import lsst.afw.image as afwImage
import lsst.afw.display as afwDisp
import lsst.afw.coord as afwCoord
import lsst.afw.geom as afwGeom

from pfs.datamodel.pfsArm import PfsArm, PfsConfig

butler = dafPersist.Butler(os.path.join("/Users/rhl/PFS/Data/rerun", "rhl", "talk"))
afwDisplay.setDefaultBackend("lsst.display.ginga" if True else "ds9")
afwDisplay.Display.delAllDisplays()
disp = afwDisplay.Display(1, open=True)
disp2 = afwDisplay.Display(2, open=True)

figDir = os.path.expanduser("~/TeX/Talks/PFS/Princeton-2016-09") if False else None

dataId = dict(visit=4, arm="r", spectrograph=2)

fiberId = 1 + np.arange(11)
ra  = np.zeros_like(fiberId, dtype='float32')
dec = np.zeros_like(ra)

catId = np.zeros_like(fiberId)
objId = np.zeros_like(catId)

filterNames = "grizy"
fiberMags   = np.zeros((len(fiberId), len(filterNames)))
mpsCentroid = np.zeros((len(fiberId), 2))

pfsConfig = PfsConfig(#tract=[0], patch=['0,0'], 
                      fiberId=fiberId, ra=ra, dec=dec,
                      catId=catId, objId=objId, fiberMag=fiberMags, filterNames=filterNames,
                      mpsCen=mpsCentroid)

if False:
    pfsConfig.write(dirName="/Users/rhl")
else:
    from pfs.drp.stella.datamodelIO import PfsConfigIO

    dateObs = butler.queryMetadata('raw', 'dateObs', dataId)[0]

    butler.put(PfsConfigIO(pfsConfig), 'pfsConfig',
               dict(dateObs=dateObs, pfsConfigId=pfsConfig.pfsConfigId))

exp = butler.get("postISRCCD", dataId, visit=4)

if False:
    disp2.mtv(exp, title="postISRCCD")

calexp = butler.get("calexp", dataId, visit=4)

disp.mtv(calexp, title="calexp")
disp.scale('linear', 'zscale')

disp.get_viewer().show()

disp2.mtv(calexp.getPsf().computeImage())
disp2.zoom(32)
disp2.pan(0, 0)

disp2.get_viewer().show()

import lsst.meas.algorithms.utils as maUtils

mos = maUtils.showPsfMosaic(calexp, nx=8, ny=8, showCenter=False, showFwhm=False, display=disp)
disp.scale('linear', 'minmax')
disp.get_viewer().show()

disp.pan(3197, 783)
disp.zoom(2)

get_ipython().magic('pdb 1')
bias = butler.get("bias", dataId, arm="r")  # arm="m" fails
disp.mtv(bias, title="bias")
disp.scale('linear', 'zscale')
disp.get_viewer().show()

dark = butler.get("dark", dataId)
disp.mtv(dark, title="dark")
disp.get_viewer().show()

if False:
    flat = butler.get("flat", dataId)
    disp.mtv(flat, title="flat")
    #disp.get_viewer().show()

raw = butler.get("raw", dataId, visit=7253)
disp.mtv(raw, title="raw 7253")

raw2 = butler.get("raw", dataId, visit=7251)
disp2.mtv(raw2, title="raw 7251")

print "Exposure time: %.1fs" % (raw.getCalib().getExptime())

disp2.get_viewer().show()

import lsst.afw.cameraGeom.utils as cgUtils

disp.mtv(raw, title='Raw dark 7291')
disp.scale('linear', 'zscale')

cgUtils.overlayCcdBoxes(raw.getDetector(), raw.getBBox(), nQuarter=0, isTrimmed=False,
                        ccdOrigin=afwGeom.PointI(0,0), display=disp, binSize=1)
disp.get_viewer().show()

visit = 7291 if True else 4
raw = butler.get("raw", dataId, visit=visit)
disp2.mtv(raw, title="dark" if visit == 7291 else "arc")

print "%.1fs" % (raw.getCalib().getExptime())
disp2.get_viewer().show()

import lsst.afw.math as afwMath
import lsst.afw.display.utils as afwDisplayUtils

spot = exp[1967:1997, 2254:2284].getMaskedImage().getImage()
spot1 = afwMath.offsetImage(spot, 0.5, 0)
spot2 = afwMath.offsetImage(spot1, -0.5, 0)
spot2 -= spot
m = afwDisplayUtils.Mosaic()
#m.makeMosaic([spot, spot1, spot2]) # , display=disp)
disp.mtv(spot2)
disp.scale('linear', 'minmax')
#disp.get_viewer().show()

pfsArm = butler.get("pfsArm", dataId)
if False:
    dateObs = butler.queryMetadata('raw', 'dateObs', dataId)[0]

    pfsArm.pfsConfig = butler.get("pfsConfig", dataId,
                                  pfsConfigId=pfsArm.pfsConfigId, dateObs=dateObs)
    pfsArm.checkPfsConfig()

pfsArm.plot()

for fid in range(1, len(pfsArm.flux[:, 0]) + 1):
    if fid == 11:
        pfsArm.plot(fid, showFlux=True, showPlot=False)
plt.xlim(600, 1000)
#plt.ylim(-10, 10000)
if figDir:
    plt.savefig(os.path.join(figDir, "arc-4-extracted.png"))
plt.show()

for i in range(len(pfsArm.flux[:, 0])):
    plt.plot(pfsArm.lam[i], pfsArm.flux[i], label=str(i + 1))
plt.xlim(856 - 10, 858 + 10)
plt.xlabel(r"$\lambda (nm)$")
plt.ylim(-10, 5000)
plt.legend(loc='upper right')
if figDir:
    plt.savefig(figDir, "arc-4-extracted-detail.png")
plt.show()

dataId = dict(visit=4, arm="r", spectrograph=2)
calexp = butler.get("postISRCCD", dataId, visit=4)

disp.mtv(calexp, title="calexp")
disp.scale('linear', 'zscale')

import lsst.afw.detection as afwDetect
import lsst.afw.display.utils as afwDisplayUtils

defects = afwDetect.FootprintSet(calexp.getMaskedImage().getMask(), 
                                 afwDetect.Threshold(afwDetect.Threshold.BITMASK)).getFootprints()

disp.erase()
with disp.Buffering():
    for d in butler.get("defects", dataId):
        afwDisplayUtils.drawBBox(d.getBBox(), display=disp,
                                 ctype=afwDisplay.GREEN, borderWidth=0.5)

raw = butler.get("raw", dataId, visit=4).convertF()
for a in raw.getDetector():
    mi = raw.getMaskedImage()
    overscanImage = mi[a.getRawHorizontalOverscanBBox()]
    mi[a.getRawBBox()].getImage()[:] -= np.median(overscanImage.getImage().getArray())
    
if True:
    disp.mtv(raw, title="raw")
else:
    disp.erase()
disp.scale('linear', 'zscale')

import lsst.afw.cameraGeom.utils as cgUtils

cgUtils.overlayCcdBoxes(raw.getDetector(), raw.getBBox(), nQuarter=0, isTrimmed=False,
                        ccdOrigin=afwGeom.PointI(0,0), display=disp, binSize=1)

with disp.Buffering():
    for a in raw.getDetector():
        delta = a.getBBox().getMin() - a.getRawBBox().getMin()
        for d in butler.get("defects", dataId):
            if a.getBBox().contains(d.getBBox()):
                afwDisplayUtils.drawBBox(d.getBBox(), display=disp, origin=delta,
                                         ctype=afwDisplay.GREEN, borderWidth=0.5)

disp.get_viewer().show()

import lsst.afw.coord as afwCoord
afwImage.makeWcs(afwCoord.Coord(afwGeom.PointD(0, 0), afwGeom.degrees), afwGeom.PointD(0, 0), 1, 0, 0, 1)

