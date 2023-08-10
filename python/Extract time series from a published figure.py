# Load image and libraries
get_ipython().magic('matplotlib inline')
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
input_image = misc.imread('figure_processed.png')

# Convert input image from RGBA to binary
input_image = input_image - 255
input_image = np.mean(input_image,2)
binary_image = input_image[::-1,:]
binary_image[binary_image>0] = 1
Npixels_rate,Npixels_time = np.shape(binary_image)

# Visualize binary image
plt.figure(figsize=(8,5))
plt.pcolor(np.arange(Npixels_time),np.arange(Npixels_rate),binary_image, cmap=cm.bone)
plt.xlim((0,Npixels_time))
plt.ylim((0,Npixels_rate))
plt.xlabel('Time (pixels)',size=20)
plt.ylabel('Firing rate (pixels)',size=20)

# Extract the time series (not the scale bars) by starting in the first column
col_in_time_series = True
s1rate_pixels = []
col = 0
while col_in_time_series == True:
    if len(np.where(binary_image[:,col]==1)[0]):
        s1rate_pixels.append(np.mean(np.where(binary_image[:,col]==1)[0]))
    else:
        col_in_time_series = False
    col += 1
s1rate_pixels = np.array(s1rate_pixels)

# Subtract baseline
s1rate_pixels = s1rate_pixels - np.min(s1rate_pixels)

# Visualize time series
plt.figure(figsize=(5,5))
plt.plot(s1rate_pixels,'k',linewidth=3)
plt.xlabel('Time (pixels)',size=20)
plt.ylabel('Firing rate (pixels)',size=20)

# Convert rate from pixels to Hz
ratescale_col = 395 # Column in image containing containing rate scale
rate_scale = 50 # Hz, scale in image
ratescale_Npixels = np.sum(binary_image[:,ratescale_col])
pixels_to_rate = rate_scale/ratescale_Npixels
s1rate = s1rate_pixels*pixels_to_rate

# Convert time from pixels to ms
timescale_row = np.argmax(np.mean(binary_image[:,400:],1)) # Row in image containing time scale
time_scale = 100 # ms, scale in image
timescale_Npixels = np.sum(binary_image[timescale_row,400:])
pixels_to_time = time_scale/timescale_Npixels
pixels = np.arange(len(s1rate_pixels))
t = pixels*pixels_to_time

# Visualize re-scaled time series
plt.figure(figsize=(5,5))
plt.plot(t, s1rate,'k',linewidth=3)
plt.xlabel('Time (ms)',size=20)
plt.ylabel('Firing rate (Hz)',size=20)

# Interpolate time series to sample every 1ms
from scipy import interpolate
f = interpolate.interp1d(t, s1rate) # Set up interpolation
tmax = np.floor(t[-1])
t_ms = np.arange(tmax) # Desired time series, in ms
s1rate_ms = f(t_ms) # Perform interpolation

# Visualize re-scaled time series
plt.figure(figsize=(5,5))
plt.plot(t_ms, s1rate_ms,'k',linewidth=3)
plt.xlabel('Time (ms)',size=20)
plt.ylabel('Firing rate (Hz)',size=20)

# Save final time series
np.save('extracted_timeseries',s1rate_ms)

