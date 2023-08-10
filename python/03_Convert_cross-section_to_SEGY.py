import numpy as np
# for plotting results
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
# for resizing and resampling
from scipy.ndimage import interpolation
# for the quantization
from sklearn.metrics import pairwise_distances_argmin
# for exporting to SEGY
from obspy.core import Trace, Stream, Stats
from obspy.io.segy.segy import SEGYBinaryFileHeader, SEGYTraceHeader
import datetime
# for displaying the plots in the notebook
get_ipython().magic('matplotlib inline')

inFile = r'..\data\Kevitsa_Titan_Line_E5_crop.png'
imRGB = plt.imread(inFile)
# plot
fig,ax = plt.subplots(figsize=(10,6))
ax.imshow(imRGB,aspect='equal')
plt.title('Original RGB image')

imRGB.shape

# I want to increase the number of rows from 704 to 1000, so:
stretch_factor = 1000./704

# Nearest-neighbour interpolation
imRGB_resized = interpolation.zoom(imRGB,zoom=(stretch_factor,1,1),order=0)
fig,ax = plt.subplots(figsize=(10,6))
ax.imshow(imRGB_resized,aspect='equal')

inFile = r'..\data\Windows_256_color_palette_RGB.csv'
win256 = np.loadtxt(inFile,delimiter=',')

nrows,ncols,d = imRGB_resized.shape
flat_array = np.reshape(imRGB_resized, (nrows*ncols, 3))
flat_array[:5]

# run function, making sure the palette data is normalised to the 0-1 interval
indices = pairwise_distances_argmin(flat_array,win256/255)
# reshape the indices to the shape of the initial image
indexedImage  = indices.reshape((nrows,ncols))

new_cm = mcolors.LinearSegmentedColormap.from_list('win256', win256/255)
plt.register_cmap(cmap=new_cm)  # optional but useful to be able to call the colormap by its name.

fig,ax = plt.subplots(figsize=(10,6))
ax.imshow(indexedImage,cmap='win256',norm=mcolors.NoNorm(),aspect='equal')

# save array to file
outFile = r'..\data\Kevitsa_Titan_Line_E5_oneband.npy'
np.save(outFile,indexedImage)

xStart = 3496937
yStart = 7509361
xEnd = 3500188
yEnd = 7514290
# create linearly interpolated sequence of points between the two ends of the line
traces = np.arange(ncols)
X = np.interp(traces,[0,ncols-1],[xStart,xEnd])
Y = np.interp(traces,[0,ncols-1],[yStart,yEnd])

dt = 1750 / nrows / 1000
dt

# convert data format
data = np.asarray(indexedImage,dtype=np.float32)
data.dtype

# create coordinate vectors with applied scaling factor
scaler = 100
Xint = np.round(X*scaler).astype(np.int)
Yint = np.round(Y*scaler).astype(np.int)
Xint.dtype

# Make a new Stream object
out = Stream() 
# For each column in the image, create a new trace
for i,t in enumerate(data.T):       
    trace = Trace(t, header={'delta': dt }) 
    out.append(trace)

# create trace headers with trace coordinates
for i,trace in enumerate(out):
    trace.stats.segy = {}
    trace.stats.segy.trace_header = SEGYTraceHeader()
    
    trace.stats.segy.trace_header.trace_sequence_number_within_line = i+1
    trace.stats.segy.trace_header.sample_interval_in_ms_for_this_trace = dt
    trace.stats.segy.trace_header.number_of_samples_in_this_trace = len(trace)
    trace.stats.segy.trace_header.scalar_to_be_applied_to_all_coordinates = -1*scaler
    trace.stats.segy.trace_header.delay_recording_time = 0
    trace.stats.segy.trace_header.source_coordinate_x= Xint[i]
    trace.stats.segy.trace_header.source_coordinate_y= Yint[i]
    trace.stats.segy.trace_header.x_coordinate_of_ensemble_position_of_this_trace= Xint[i]
    trace.stats.segy.trace_header.y_coordinate_of_ensemble_position_of_this_trace=Yint[i]

# create text header
text_header = '{0:<80}'.format("File created on "+datetime.date.today().isoformat())
text_header += '{0:<80}'.format("Coordinates of the line:")
text_header += '{0:<80}'.format("LeftX   : "+ str(int(X[0])))
text_header += '{0:<80}'.format("LeftY   : "+ str(int(Y[0])))
text_header += '{0:<80}'.format("RightX  : "+ str(int(X[-1])))
text_header += '{0:<80}'.format("RightY  : "+ str(int(Y[-1])))

# Add text and binary headers to stream
out.stats = Stats(dict(textual_file_header=text_header.encode('utf-8')))
out.stats.binary_file_header = SEGYBinaryFileHeader()
out.stats.binary_file_header.number_of_data_traces_per_ensemble = 1
out.stats.binary_file_header.number_of_samples_per_data_trace = len(trace)

# create SEGY file
outFile = r'..\data\Kevitsa_Titan_Line_E5.sgy'
print("\nCreating " + outFile)
out.write(outFile, format='SEGY', data_encoding=5) 
print("Done!")



