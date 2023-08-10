import ftplib

ftp = ftplib.FTP('podaac-ftp.jpl.nasa.gov')
ftp.login()
ftp.cwd('allData/rapidscat/L2B12/v1.3/2016/232')
ftp.retrbinary('RETR rs_l2b_v1.3_10827_201609290531.nc.gz', open('ISS.nc.gz', 'wb').write)
ftp.quit()

get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import netCDF4 as nc

f = nc.Dataset('ISS.nc', 'r')
print(f.variables.keys())

z = f.variables['retrieved_wind_speed']

z.dimensions

fig = plt.figure(figsize = (20,20))
ax = fig.add_subplot(111)
img = ax.imshow(f.variables['retrieved_wind_speed'][:].transpose(), interpolation=None)

