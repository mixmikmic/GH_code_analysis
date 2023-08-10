from owslib.wcs import WebCoverageService

my_wcs = WebCoverageService('http://earthserver.pml.ac.uk/rasdaman/ows?', version='2.0.0')

# print out coverages that have CCI in title
for coverage_name in my_wcs.contents.keys():
    if 'CCI' in coverage_name:
        print coverage_name

print my_wcs.contents['CCI_V2_release_chlor_a']

for item in dir(my_wcs.contents['CCI_V2_release_chlor_a']):
    if "_" not in item:
        print item

my_wcs.contents['CCI_V2_release_chlor_a'].boundingboxes

for item in dir(my_wcs.contents['CCI_V2_release_chlor_a'].grid):
    if "_" not in item:
        print item + ": " + str(my_wcs.contents['CCI_V2_release_chlor_a'].grid.__dict__[item])

my_wcs.contents['CCI_V2_release_chlor_a'].supportedFormats

for time in my_wcs.contents['CCI_V2_release_chlor_a'].timepositions[0:10]:
    print time.isoformat()
    

get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import netCDF4 as nc


coverage_file = my_wcs.getCoverage(identifier=['CCI_V2_monthly_chlor_a'], format='application/netcdf', subsets=[('Long',-10,0), ('Lat',40,50),('ansi',"2000-07-31T00:00:00","2000-08-31T00:00:00")])

with open('testout.nc', 'w') as outfile:
    outfile.write(coverage_file.read())
    
ncdata = nc.Dataset("testout.nc", "a", format="NETCDF4")
ncdata.variables['chlor_a'].setncattr('missing_value', 9.969209968386869e+36)
data = np.flipud(np.rot90(ncdata.variables['chlor_a'][:,:,0]))
print data.shape



plt.figure(figsize=(40,20))
plt.imshow(data,norm=colors.LogNorm()) 



