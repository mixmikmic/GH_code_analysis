get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import scipy.signal as signal
import scipy.io as sio
import scipy as sp
import numpy as np
import math
import matplotlib.pyplot as plt
from find_grid_cells import load_data
from calculate_2d_tuning_curve import calculate_2d_tuning_curve
from calculate_spatial_periodicity import calculate_spatial_periodicity
from calculate_correlation_matrix import calculate_correlation_matrix
from analyze_periodicity import analyze_periodicity, Partition_Data
from fourier_transform import (fourier_transform, analyze_fourier, analyze_polar_spectrogram, 
    analyze_fourier_rings, fourier_rings_significance)
from shuffle_rate_map import shuffle_rate_map
from plot_data import (plot_canonical_scoring_method, plot_analyze_correlation_periodicity, plot_fourier_scoring,  
    plot_polar_components, plot_rho_mean_power, plot_ring_power, plot_ring_random_distribution)

boxSize = 100
nPosBins = 20
filePath = './data/11343-08120502_t8c2.mat'

files = ['11343-08120502_t8c2.mat', '11207-21060503_t8c1.mat']

posx1, posy1, t1, dt1, spiketrain1, smoothFiringRate1 = load_data('./data/11343-08120502_t8c2.mat')
posx2, posy2, t2, dt2, spiketrain2, smoothFiringRate2 = load_data('./data/11207-21060503_t8c1.mat')
#posx1, posy1, t1, dt1, spiketrain1, smoothFiringRate1 = load_data('./data/11138-11040509_t5c1.mat')
#posx1, posy1, t1, dt1, spiketrain1, smoothFiringRate1 = load_data('./data/11207-11060502_t6c2.mat')
#posx2, posy2, t2, dt2, spiketrain2, smoothFiringRate2 = load_data('./data/11207-21060501+02_t6c1.mat')
#posx2, posy2, t2, dt2, spiketrain2, smoothFiringRate2 = load_data('./data/11084-10030502_t1c2.mat')

# Calculates the unsmoothed and smoothed rate map
# Smoothed rate map used for correlation calculations, unsmoothed rate map used for fourier analysis
unsmoothRateMap1, smoothRateMap1 = calculate_2d_tuning_curve(posx1, posy1, smoothFiringRate1, nPosBins, 0, boxSize)
unsmoothRateMap2, smoothRateMap2 = calculate_2d_tuning_curve(posx2, posy2, smoothFiringRate2, nPosBins, 0, boxSize)

combineUnsmoothRateMap = unsmoothRateMap1 + unsmoothRateMap2

def gaussian_2d(shape=(3, 3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

h = gaussian_2d()
combineSmoothRateMap = sp.ndimage.correlate(combineUnsmoothRateMap, h, mode='constant')

# Calculates the correlation matrix from the smooted rate map
correlationMatrix = calculate_correlation_matrix(combineSmoothRateMap)

# Determines the spatial periodicity of the correlation matrix by calculating the correlation of the 
# matrix in intervals of 6 degrees 
rotations, correlations, gridScore, circularMatrix, threshold = calculate_spatial_periodicity(correlationMatrix)

plot_canonical_scoring_method(combineSmoothRateMap, correlationMatrix, rotations, correlations, gridScore, circularMatrix, threshold)

# Partitions the correlation periodicity curve into 3-10 periods and 
# collapses/sums the partitioned data 
collapsePartitionData, maxCollapseValues = analyze_periodicity(rotations, correlations)

plot_analyze_correlation_periodicity(collapsePartitionData, maxCollapseValues)

spiketrain1 = np.reshape(spiketrain1, (len(spiketrain1), 1))
spiketrain2 = np.reshape(spiketrain2, (len(spiketrain2), 1))
combineSpiketrain = np.vstack((spiketrain1, spiketrain2))

combinePosx = np.vstack((posx1, posx2))
combinePosy = np.vstack((posy1, posy2))

# Calculates the two-dimensional Fourier spectrogram
adjustedRateMap = combineUnsmoothRateMap - np.nanmean(combineUnsmoothRateMap)
meanFr = np.sum(combineSpiketrain) / (t1[-1] + t2[-1])
meanFr = meanFr[0]
fourierSpectrogram, polarSpectrogram, beforeMaxPower, maxPower, isPeriodic = fourier_transform(adjustedRateMap, meanFr, combineSpiketrain, dt1, combinePosx, combinePosy)

print('Combined Fourier and Polar Spectrogram')
plot_fourier_scoring(fourierSpectrogram, polarSpectrogram)

adjustedRateMap1 = unsmoothRateMap1 - np.nanmean(unsmoothRateMap1)
meanFr1 = np.sum(spiketrain1) / t1[-1]
meanFr1 = meanFr1[0]
fourierSpectrogram1, polarSpectrogram1, beforeMaxPower1, maxPower1, isPeriodic1 = fourier_transform(adjustedRateMap1, meanFr1, spiketrain1, dt1, posx1, posy1)

print('Cell 1 Fourier and Polar Spectrogram')
plot_fourier_scoring(fourierSpectrogram1, polarSpectrogram1)

adjustedRateMap2 = unsmoothRateMap2 - np.nanmean(unsmoothRateMap2)
meanFr2 = np.sum(spiketrain2) / t2[-1]
meanFr2 = meanFr2[0]
fourierSpectrogram2, polarSpectrogram2, beforeMaxPower2, maxPower2, isPeriodic2 = fourier_transform(adjustedRateMap2, meanFr2, spiketrain2, dt2, posx2, posy2)

print('Cell 2 Fourier and Polar Spectrogram')
plot_fourier_scoring(fourierSpectrogram2, polarSpectrogram2)

polarComponents = analyze_fourier(fourierSpectrogram, polarSpectrogram)

plot_polar_components(polarComponents)

rhoMeanPower, localMaxima = analyze_polar_spectrogram(polarSpectrogram)
    
plot_rho_mean_power(rhoMeanPower, localMaxima)

# Area of ring set to 1605 for a 256x256 matrix
area = 1605
maxRadius = math.floor(math.sqrt((fourierSpectrogram.shape[1]/2)**2 + (fourierSpectrogram.shape[0]/2)**2))

averageRingPower, radii = analyze_fourier_rings(fourierSpectrogram, area)

plot_ring_power(averageRingPower, radii, maxRadius, 'Combined Cell 1 and Cell 2')

averageRingPower1, radii1 = analyze_fourier_rings(fourierSpectrogram1, 1605)

plot_ring_power(averageRingPower1, radii1, maxRadius, 'Cell 1')

averageRingPower2, radii2 = analyze_fourier_rings(fourierSpectrogram2, 1605)

plot_ring_power(averageRingPower2, radii2, maxRadius, 'Cell 2')

maxRadius = math.floor(math.sqrt((256/2)**2 + (256/2)**2))

randomDistribution1, confidenceInterval1, shuffleRateMap1, shuffleFourier1 = fourier_rings_significance(combineUnsmoothRateMap, combineSpiketrain, t1, t2, dt1, combinePosx, combinePosy, 'rate map', 1)
plot_ring_random_distribution(randomDistribution1, averageRingPower, radii, confidenceInterval1, shuffleRateMap1, shuffleFourier1, maxRadius, 1)

randomDistribution2, confidenceInterval2, shuffleRateMap2, shuffleFourier2 = fourier_rings_significance(combineUnsmoothRateMap, combineSpiketrain, t1, t2, dt1, combinePosx, combinePosy, 'rate map', 2)
plot_ring_random_distribution(randomDistribution2, averageRingPower, radii, confidenceInterval2, shuffleRateMap2, shuffleFourier2, maxRadius, 2)

randomDistribution3, confidenceInterval3, shuffleRateMap3, shuffleFourier3 = fourier_rings_significance(combineUnsmoothRateMap, combineSpiketrain, t1, t2, dt1, combinePosx, combinePosy, 'rate map', 4)
plot_ring_random_distribution(randomDistribution3, averageRingPower, radii, confidenceInterval3, shuffleRateMap3, shuffleFourier3, maxRadius, 4)

randomDistribution4, confidenceInterval4, shuffleRateMap4, shuffleFourier4 = fourier_rings_significance(combineUnsmoothRateMap, combineSpiketrain, t1, t2, dt1, combinePosx, combinePosy, 'rate map', 5)
plot_ring_random_distribution(randomDistribution4, averageRingPower, radii, confidenceInterval4, shuffleRateMap4, shuffleFourier4, maxRadius, 5)

randomDistribution5, confidenceInterval5, shuffleRateMap5, shuffleFourier5 = fourier_rings_significance(combineUnsmoothRateMap, combineSpiketrain, t1, t2, dt1, combinePosx, combinePosy, 'rate map', 10)
plot_ring_random_distribution(randomDistribution5, averageRingPower, radii, confidenceInterval5, shuffleRateMap5, shuffleFourier5, maxRadius, 10)

randomDistributionAverage = (randomDistribution1 + randomDistribution2 + randomDistribution3 + randomDistribution4 + randomDistribution5) / 5

maxRadius = math.floor(math.sqrt((256/2)**2 + (256/2)**2))
radii = np.arange(0, maxRadius)
plt.plot(radii, randomDistributionAverage, label="Average random distribution")
plt.plot(radii, averageRingPower, label="Average power distribution")
ax = plt.gca()
plt.xlabel('Inner Radius Length')
plt.ylabel('Average Power')
plt.xlim(0, maxRadius)
plt.legend()
plt.show()

randomDistributionAverage = np.reshape(randomDistributionAverage, (len(randomDistributionAverage), 1))
difference = averageRingPower - randomDistributionAverage

plt.plot(radii, difference)
plt.title('Difference between average ring power and random distribution power')
plt.xlabel('Inner Radius Length')
plt.ylabel('Difference')
plt.xlim(0, maxRadius)
plt.axhline(0, color='black')
plt.show()

