import urllib # import the module urllib

url = "https://nomads.ncdc.noaa.gov/data/gfs4/201702/20170201/" # Link of the directory with the files to download
outpath = "." # Define the directory where you want to store the data

filename = "gfs_4_20170201_0000_000.grb2"  
print url+filename # print the full link of the file

urllib.urlretrieve(url+filename, outpath+filename) # arg1: link of the file, arg2: output path 

rng = range(1, 121)
print rng

for r in rng:
    print str(r).zfill(3)

basename = "gfs_4_20170201_0000_"

basename + str(r).zfill(3)+".grb2" # full filename

filenames = [basename + str(r).zfill(3)+".grb2" for r in rng]
for filename in filenames:
    print filename

for filename in filenames:
    print filename
    urllib.urlretrieve(url+filename, outpath+filename)



