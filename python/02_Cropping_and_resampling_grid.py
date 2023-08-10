import numpy as np
from scipy import interpolate
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
get_ipython().magic('matplotlib inline')

indexedImage = np.load(r'..\data\Kevitsa_geology_indexed.npy')
# load colormap
win256 = np.loadtxt(r'..\data\Windows_256_color_palette_RGB.csv',delimiter=',')
new_cm = mcolors.LinearSegmentedColormap.from_list('win256', win256/255)
# make plot
fig,ax = plt.subplots(figsize=(6,6))
ax.imshow(indexedImage,cmap=new_cm,norm=mcolors.NoNorm())

inFile = r'..\data\Kevitsa_geology_noframe.png'
dataset = rasterio.open(inFile)

nrows,ncols = dataset.shape
geotransform = dataset.get_transform()
cellsize = geotransform[1] 
# top-left corner of top-left pixel
ulx =      geotransform[0] 
uly =      geotransform[3]
# bottom-right corner of bottom-right pixel
lrx = ulx + geotransform[1] * ncols
lry = uly + geotransform[5] * nrows

# print the output
print('Raster Size: {:d} columns x {:d} rows x {:d} bands'.format(ncols, nrows, dataset.count))
print('Cell Size: {:.2f} m'.format(cellsize))
print('Upper-left corner:({:.2f},{:.2f})\nBottom-right corner:({:.2f},{:.2f})'.format(ulx,uly,lrx,lry))

xmin = ulx + cellsize/2.
ymin = lry + cellsize/2.
xmax = xmin + (ncols-1)*cellsize
ymax = ymin + (nrows-1)*cellsize

# 1-D arrays of coordinates (use linspace to avoid errors due to floating point rounding)
x = np.linspace(xmin,xmax,num=ncols,endpoint=True)
y = np.linspace(ymin,ymax,num=nrows,endpoint=True)
# 2-D arrays of coordinates
X,Y = np.meshgrid(x,y)

Y = np.flipud(Y)

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(8,6))
ax1.imshow(X)
ax1.set_title('X')
ax2.imshow(Y)
ax2.set_title('Y')
plt.show()

interp = interpolate.NearestNDInterpolator(np.column_stack((X.flatten(),Y.flatten())),indexedImage.flatten())

# First let's define a grid of in-line and cross-line indices
pad = 50  # number of rows and columns to add on all sides
step = 1  # 
inline_limits = np.arange(1000-pad,1280+pad,step)
xline_limits = np.arange(1000-pad,1287+pad,step)
inline,xline = np.meshgrid(inline_limits,xline_limits,indexing='ij')
# indexing starts from bottom-left corner
inline = np.flipud(inline)

# Now we can compute the coordinates - these numbers come from the "advanced" panel in the coordinate settings in OpendTect
Xi = 3491336.248 - 3.19541219*inline + 9.4758042*xline
Yi = 7497848.4 + 9.47383513*inline + 3.19552448*xline

newImage = interp((Xi,Yi))
plt.imshow(newImage,cmap=new_cm)

# Trace indices arranged in two vectors
IL_vector = inline.flatten()
XL_vector = xline.flatten()

# Data arranged in one single vector
values = newImage.flatten()

# Z column (zeros for importing in OpendTect)
Z_vector = np.zeros_like(IL_vector)

# Save to ASCII
outFile = r'..\data\Kevitsa_geology_indexed_ILXL.xyz'
np.savetxt(outFile,np.array((IL_vector,XL_vector,Z_vector,values)).T,fmt='%14.2f %14.2f %14.2f %.8g')
print('Done!')

