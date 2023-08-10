import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pandas as pd

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import diffimTests as dit
#reload(dit)

testObj = dit.DiffimTest(n_sources=2000)
exposure = testObj.im1.asAfwExposure()

import lsst.ip.diffim as ipDiffim
#reload(ipDiffim)
config = ipDiffim.ImageMapReduceConfig()

task = ipDiffim.ImageMapReduceTask(config=config)
plt.figure(figsize=(10,10))
task._generateGrid(exposure, forceEvenSized=True)
task.plotBoxes(exposure.getBBox(), skip=3);

newExp = task.run(exposure).exposure  # this just returns the same exposure

print dit.computeClippedImageStats(exposure.getMaskedImage().getImage().getArray())
print dit.computeClippedImageStats(newExp.getMaskedImage().getImage().getArray())
print dit.computeClippedImageStats(exposure.getMaskedImage().getImage().getArray()-newExp.getMaskedImage().getImage().getArray())
dit.plotImageGrid((exposure.getMaskedImage().getImage(), newExp.getMaskedImage().getImage()), clim=(-50,150), imScale=8)

import lsst.ip.diffim as ipDiffim
#reload(ipDiffim)
config = ipDiffim.ImageMapReduceConfig()
config.gridStepX = config.gridStepY = 8

task = ipDiffim.ImageMapReduceTask(config=config)
plt.figure(figsize=(10,10))
task._generateGrid(exposure, forceEvenSized=True)
task.plotBoxes(exposure.getBBox(), skip=5);

newExp = task.run(exposure).exposure  # this just returns the exposure

print dit.computeClippedImageStats(exposure.getMaskedImage().getImage().getArray())
print dit.computeClippedImageStats(newExp.getMaskedImage().getImage().getArray())
print dit.computeClippedImageStats(exposure.getMaskedImage().getImage().getArray()-newExp.getMaskedImage().getImage().getArray())
dit.plotImageGrid((exposure.getMaskedImage().getImage(), newExp.getMaskedImage().getImage()), clim=(-50,150), imScale=8)



