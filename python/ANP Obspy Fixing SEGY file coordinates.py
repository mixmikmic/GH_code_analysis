get_ipython().magic('cd data')
stream = utils.obspy.read('R0003_BAHIA_SUL.0231-1496.MIG_FIN.93.sgy', format="SEGY")

get_ipython().magic('cd /media/andre/KINGDRE/Bahia_Sul')
from obspy.segy.core import readSEGY
section = readSEGY('R0003_BAHIA_SUL.0232-0157.MIG_FIN.35.sgy', unpack_trace_headers=True)

import obspy.segy.header as segyheaders
segyheaders.TRACE_HEADER_FORMAT # trace header keys per byte definition, index zero based (%!definition table!%)
section.stats.textual_file_header  # from the segy loaded
section.stats.binary_file_header.keys() # from the segy loaded

print section

import numpy, pylab
from matplotlib import pyplot
from scipy.interpolate import InterpolatedUnivariateSpline
from obspy.segy.core import readSEGY
# getting just not repeated coordinate values
ntr = len(section)
sx = numpy.zeros(1)
sy = numpy.zeros(1)
trc = numpy.zeros(1) # trace index
cdpx = numpy.zeros(ntr)
cdpy = numpy.zeros(ntr)
# 181, 185 cdpx, cdpy (first point allways in)
cdpx[0] = section[0].stats.segy.trace_header.x_coordinate_of_ensemble_position_of_this_trace
cdpy[0] = section[0].stats.segy.trace_header.y_coordinate_of_ensemble_position_of_this_trace
sx[0] = cdpx[0] 
sy[0] = cdpy[0]
trc[0] = 0
for i in numpy.arange(1, ntr):
    cdpx[i] = section[i].stats.segy.trace_header.x_coordinate_of_ensemble_position_of_this_trace
    cdpy[i] = section[i].stats.segy.trace_header.y_coordinate_of_ensemble_position_of_this_trace
    if (cdpx[i] != cdpx[i-1]) or (cdpy[i] != cdpy[i-1]) : # just in case (x, y) == (x, y) ignore
        sx = numpy.append(sx, cdpx[i])    
        sy = numpy.append(sy, cdpy[i])
        trc = numpy.append(trc, i)
pylab.rcParams['figure.figsize'] = 12, 6 
print len(trc), len(sx), len(sy), len(cdpx), len(cdpy), 
pyplot.plot(trc[:30], sx[:30], '^w', numpy.arange(trc[30]), cdpx[:trc[30]], '+')
pyplot.legend(['not duplicated', 'all'], loc='best')
pyplot.show()

from scipy.interpolate import InterpolatedUnivariateSpline
x = trc # not duplicated indexes
y = sx # not duplicated coordinates
flinear = InterpolatedUnivariateSpline(x, y, bbox=[-3, ntr+2], k=1) # linear iterp function case where spline degree k=1
xnew = numpy.arange(0, ntr, 1) # indexes of all traces 
pyplot.plot(trc[:30], sx[:30], '^w', numpy.arange(trc[30]), cdpx[:trc[30]], '+', xnew[:trc[30]], flinear(xnew[:trc[30]]), 'k.')
pyplot.legend(['not duplicated', 'all', 'corrected'], loc='best')
pyplot.show()

from scipy.interpolate import InterpolatedUnivariateSpline
from obspy.segy.core import readSEGY
section = readSEGY('R0003_BAHIA_SUL.0231-1496.MIG_FIN.93.sgy', unpack_trace_headers=True)
# getting just not repeated coordinate values sx, sy
ntr = len(section) # number of traces
sx = numpy.zeros(1) 
sy = numpy.zeros(1)
trc = numpy.zeros(1) # trace index of not duplicated traces
cdpx = numpy.zeros(ntr) # original x coordinate
cdpy = numpy.zeros(ntr) # original y coordinate
# bytes (181, 185) (cdpx, cdpy) (first point allways in)
cdpx[0] = section[0].stats.segy.trace_header.x_coordinate_of_ensemble_position_of_this_trace
cdpy[0] = section[0].stats.segy.trace_header.y_coordinate_of_ensemble_position_of_this_trace
sx[0] = cdpx[0] 
sy[0] = cdpy[0]
trc[0] = 0
for i in numpy.arange(1, ntr): # get just the not duplicated coordinates
    cdpx[i] = section[i].stats.segy.trace_header.x_coordinate_of_ensemble_position_of_this_trace
    cdpy[i] = section[i].stats.segy.trace_header.y_coordinate_of_ensemble_position_of_this_trace
    if (cdpx[i] != cdpx[i-1]) or (cdpy[i] != cdpy[i-1]) : # just in case (x, y) == (x, y) ignore
        sx = numpy.append(sx, cdpx[i])    
        sy = numpy.append(sy, cdpy[i])
        trc = numpy.append(trc, i)
#trc (not duplicated indexes = x)
#sx, sy not duplicated coordinates
flinearsx = InterpolatedUnivariateSpline(trc, sx, bbox=[-3, ntr+2], k=1) # linear iterp function on xcoordinate ; x is trace index
flinearsy = InterpolatedUnivariateSpline(trc, sy, bbox=[-3, ntr+2], k=1) # linear iterp function on ycoordinate ; x is trace index
# (to enable linear extrapolation that interp1 doesn't do) spline=linear iterp function case where spline degree k=1
# uses limits of extrapolation +3 traces before and after
for trace_index in numpy.arange(0, ntr, 1): # interpolate for all trace indexes, changing the trace headers on bytes (73, 77)    
    section[trace_index].stats.segy.trace_header.source_coordinate_x = int(flinearsx(trace_index))
    section[trace_index].stats.segy.trace_header.source_coordinate_y = int(flinearsy(trace_index))      
    
section.write('R0003_BAHIA_SUL.0231-1496.MIG_FIN.93B.sgy', format='SEGY')

# QC of the script
ntr = len(section)
sx = numpy.zeros(1)
sy = numpy.zeros(1)
trc = numpy.zeros(1) # trace index
cdpx = numpy.zeros(ntr)
cdpy = numpy.zeros(ntr)
# 181, 185 cdpx, cdpy (first point allways in)
cdpx[0] = section[0].stats.segy.trace_header.source_coordinate_x
cdpy[0] = section[0].stats.segy.trace_header.source_coordinate_x
sx[0] = cdpx[0] 
sy[0] = cdpy[0]
trc[0] = 0
for i in numpy.arange(1, ntr):
    cdpx[i] = section[i].stats.segy.trace_header.source_coordinate_x
    cdpy[i] = section[i].stats.segy.trace_header.source_coordinate_x
    if (cdpx[i] != cdpx[i-1]) or (cdpy[i] != cdpy[i-1]) : # just in case (x, y) == (x, y) ignore
        sx = numpy.append(sx, cdpx[i])    
        sy = numpy.append(sy, cdpy[i])
        trc = numpy.append(trc, i)
pylab.rcParams['figure.figsize'] = 12, 6 
print len(sx), len(cdpx), len(sy), len(cdpy), len(trc)
pyplot.plot(trc[:30], sx[:30], '^w', numpy.arange(trc[30]), cdpx[:trc[30]], '+')
pyplot.legend(['not duplicated', 'all'], loc='best')

segyheaders.TRACE_HEADER_FORMAT 



