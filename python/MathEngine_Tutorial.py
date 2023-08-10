# Libraries for manipulating the data
import numpy as np
import matplotlib.pyplot as plt

# Data source
deviceSerial = 'yourSerial'
inSensor     = 'yourSensorName'
inChannel    = 'yourChannelName'
outChannel   = inSensor + '_scaled'

# Uncomment the next two lines to specify start and end time for the data you want
#start = timestampInSeconds *  NANO_PER_SEC
#end = timestampInSeconds * NANO_PER_SEC

repo = TimeSeriesRepo(deviceSerial)

inSeries  = repo.getAllTimeSeries(inSensor, inChannel)
print len(inSeries), "series found"

series = inSeries[0]
values = series.getData()
timestamps= series.getTimeStamps()

print timestamps[0]
print values[0]

# We cast values to a numpy array so we can perform math operations across all of the values
# Python will treat the numpy array as it would a standard list
array = np.array(values)

# Scale the values by 10
array = array*10
print "Scaled %s points..." % len(array)

# Zip them back up with the timestamps
outData = zip(timestamps, array)

outData2 = []
for point in series:
    newPoint = (point[0],point[1]*10)
    outData2.append(newPoint)

outSeries = repo.createTimeSeries(inSensor, outChannel, series.getSampleRate(), series.getSampleRateType())

outSeries.push(outData)

outSeries.save()

# Tag the channel as a MathEngine channel so it will be deletable in the SensorCloud interface
outSeries.tagAsMathengine()

# Set the plot size and quality
matplotlib.rcParams['figure.figsize'] = (8,5)
matplotlib.rcParams['savefig.dpi'] = 100

# X axis is 0 - n
arraySize = len(array)
xValues = np.arange(arraySize)

# Maximum value in the y axis
maxY = np.amax(array)*1.25

# Plot the values and fill in the space below them
plt.plot(xValues, array, color = 'k')
plt.fill_between(xValues, array, color = '#3779ca', alpha = 0.7)

# Set the bounds of the plot
# [xMin, xMax, yMin, yMax]
plt.axis([0, arraySize, 0, maxY])

# Turn on the grid
plt.grid(True)

# Title the plot and axis
plt.title( 'Sensor: ' + inSensor +  '\n' + 
           'Channel: ' + inChannel + '\n' +
           'Sampled at: ' + str(series.getSampleRate()) + ' ' + series.getSampleRateType())
plt.xlabel( 'X Axis' )
plt.ylabel( 'Y Axis' )



