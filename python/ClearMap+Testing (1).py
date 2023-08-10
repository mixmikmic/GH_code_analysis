get_ipython().magic('matplotlib')
import os
import ClearMap.IO as io
import ClearMap.Settings as settings
filename = os.path.join(settings.ClearMapPath, 'data/template_25.tif')

import ClearMap.Visualization.Plot as clrplt

# data = io.readData(filename);
# clrplt.plotTiling(data);
#'/root/plotter.py'

#import subprocess
#with open("/root/output.png", "w+") as output:
#    subprocess.call(["python", "/root/plotter.py"], stdout=output);

data = io.readData(filename);
#clrplt.plotTiling(data);
#clrplt.plotTiling(data, inverse = True);

#Tried to hack the code by manually editing the backend plot generation to save images. Didn't work'

# from PIL import Image

# img = Image.open('/root/output.png')
# img.show() */

import ClearMap.ImageProcessing.BackgroundRemoval as bgr
dataBGR = bgr.removeBackground(data.astype('float'), size=(3,3), verbose = True);
#plt.plotTiling(dataBGR, inverse = True);

from ClearMap.ImageProcessing.Filter.DoGFilter import filterDoG
dataDoG = filterDoG(dataBGR, size=(8,8,4), verbose = True);
#plt.plotTiling(dataDoG, inverse = True, z = (10,16));

from ClearMap.ImageProcessing.MaximaDetection import findExtendedMaxima
dataMax = findExtendedMaxima(dataDoG, hMax = None, verbose = True, threshold = 10);
#plt.plotOverlayLabel( dataDoG / dataDoG.max(), dataMax.astype('int'))

from ClearMap.ImageProcessing.MaximaDetection import findCenterOfMaxima
cells = findCenterOfMaxima(data, dataMax);
print cells.shape

#plt.plotOverlayPoints(data, cells)

from ClearMap.ImageProcessing.CellSizeDetection import detectCellShape
dataShape = detectCellShape(dataDoG, cells, threshold = 15);
#plt.plotOverlayLabel(dataDoG / dataDoG.max(), dataShape, z = (10,16))

from ClearMap.ImageProcessing.CellSizeDetection import findCellSize, findCellIntensity
cellSizes = findCellSize(dataShape, maxLabel = cells.shape[0]);
cellIntensities = findCellIntensity(dataBGR, dataShape, maxLabel = cells.shape[0]);

get_ipython().magic('pyplot')
import matplotlib.pyplot as mpl
mpl.figure()
mpl.plot(cellSizes, cellIntensities, '.')
mpl.xlabel('cell size [voxel]')
mpl.ylabel('cell intensity [au]')



