import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pandas as pd

import diffimTests as dit
#reload(dit)

testObj = dit.DiffimTest(n_sources=2000)
exposure = testObj.im1.asAfwExposure()

import lsst.ip.diffim as ipDiffim
#reload(ipDiffim)
config = ipDiffim.ImageGridderConfig()

task = ipDiffim.ImageGridderTask(config=config)
plt.figure(figsize=(10,10))
task._plotBoxes(exposure);

newExp = task.run(exposure)  # this just adds 100 to the exposure

print dit.computeClippedImageStats(exposure.getMaskedImage().getImage().getArray())
print dit.computeClippedImageStats(newExp.getMaskedImage().getImage().getArray())
print dit.computeClippedImageStats(exposure.getMaskedImage().getImage().getArray()-newExp.getMaskedImage().getImage().getArray())
dit.plotImageGrid((exposure.getMaskedImage().getImage(), newExp.getMaskedImage().getImage()), clim=(-50,150), imScale=8)

import lsst.ip.diffim as ipDiffim
#reload(ipDiffim)
config = ipDiffim.ImageGridderConfig()
config.gridStepX = config.gridStepY = 8

task = ipDiffim.ImageGridderTask(config=config)
plt.figure(figsize=(10,10))
task._plotBoxes(exposure);

newExp = task.run(exposure)  # this just adds 100 to the exposure

print dit.computeClippedImageStats(exposure.getMaskedImage().getImage().getArray())
print dit.computeClippedImageStats(newExp.getMaskedImage().getImage().getArray())
print dit.computeClippedImageStats(exposure.getMaskedImage().getImage().getArray()-newExp.getMaskedImage().getImage().getArray())
dit.plotImageGrid((exposure.getMaskedImage().getImage(), newExp.getMaskedImage().getImage()), clim=(-50,150), imScale=8)



