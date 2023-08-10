import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import lsst.daf.persistence as dafPersist
import lsst.afw.cameraGeom.utils as cgUtils

import lsst.afw.display as afwDisplay

disp1 = afwDisplay.Display(1, 'ds9')
disp2 = afwDisplay.Display(2, 'ds9')

butler = dafPersist.Butler("/datasets/comCam/repo")
camera = butler.get("camera")

butler.queryMetadata("raw", ["testType"], run='4947D')

keys = ("visit", "ccd",)
dataIds = [dict(zip(keys, _)) for _ in butler.queryMetadata("raw", keys, run='4947D', testType='LAMBDA')]

for dataId in dataIds:
    if dataId["ccd"] != 'S11':
        continue
    expTime = butler.get('raw_visitInfo', dataId).getExposureTime()
    if expTime > 10 and expTime < 50:
        if butler.get('raw_md', dataId).get("MONOCH-WAVELENG") > 950:
            break

raw = butler.get('raw', dataId, ccd='S22')
#disp2.mtv(raw)

from lsst.ip.isr import AssembleCcdTask

config = AssembleCcdTask.ConfigClass()
config.doTrim = False

assembleTask = AssembleCcdTask(config=config)

exposure = assembleTask.assembleCcd(raw)
if not True:
    disp2.mtv(exposure)
else:
    disp2.erase()
cgUtils.overlayCcdBoxes(exposure.getDetector(), display=disp2, isTrimmed=config.doTrim) # , ignoreBBoxes=["ccd", "raw"])

imageSource = cgUtils.ButlerImage(butler, type='raw', verbose=True, 
                                  callback=lambda im, ccd, imageSource : 
                                           cgUtils.rawCallback(im, ccd, imageSource, subtractBias=True),
                                  isTrimmed=True, dataId=dataId)

cgUtils.showCamera(camera, imageSource=imageSource, display=disp1, showWcs=False, binSize=1); None

if True:
    mos = cgUtils.showCamera(camera, overlay=True, display=disp1)
else:
    mos = cgUtils.showCcd(camera[1], overlay=True, display=disp1)

cgUtils.plotFocalPlane(comCam)

import lsst.afw.image as afwImage
for ccd in ['S%d%d' % (i, j) for i in range(3) for j in range(3)]:
    print ccd, afwImage.readMetadata(butler.get('raw_filename', dataId, ccd=ccd)[0], 0).toOrderedDict()["LSST_NUM"]

afwImage.readMetadata(butler.get('raw_filename', dataId, ccd=ccd)[0], 0).toOrderedDict()

