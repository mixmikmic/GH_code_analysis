import numpy
import gsw

z = [-10., -50., -125., -250., -600., -1000.]
lat = 4.
gsw.p_from_z(z, lat)

z = [-10., -50., -125., -250., -600., -1000.]
lat = [4., 4., 4., 4., 4., 4.]
gsw.p_from_z(z, lat)

z = numpy.ones((3,4))
lat = numpy.ones((3,4))
gsw.p_from_z(z, lat)



