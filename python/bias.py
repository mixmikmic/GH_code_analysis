# import reader module from sparkxarray
from sparkxarray import reader
from pyspark.sql import SparkSession

# Create sparksession
spark = SparkSession.builder.appName("bias").getOrCreate()
sc = spark.sparkContext

FILE_1 = "/home/abanihi/Documents/Github/spark-xarray/datasets/AFRICA_KNMI-RACMO2.2b_CTL_ERAINT_MM_50km_1989-2008_tasmax.nc"
FILE_2 = "/home/abanihi/Documents/Github/spark-xarray/datasets/AFRICA_UC-WRF311_CTL_ERAINT_MM_50km-rg_1989-2008_tasmax.nc"

knmi = reader.ncread(sc, FILE_1, mode='single', partition_on=['rlat', 'rlon'], partitions=500, decode_times=False)

knmi.first()

wrf = reader.ncread(sc, FILE_2, mode='single', partition_on=['rlat', 'rlon'], partitions=500, decode_times=False)

wrf.first()

get_ipython().magic('time wrf.count()')

def create_indices(element):
    lat = round(float(element.rlat.data), 1)
    lon = round(float(element.rlon.data), 1)
    key = (lat, lon)
    return (key, element)

knmi2 = knmi.map(create_indices)
knmi2.first()

wrf2 = wrf.map(create_indices)
wrf2.first()

rdd = wrf2.join(knmi2, numPartitions=500)
rdd.first()

rdd.getNumPartitions()

rdd.count()

a = rdd.first()
a

def bias_correct(element):
    import numpy as np
    obs = element[1][1].tasmax.values.ravel()
    mod = element[1][0].tasmax.values.ravel()
    
    cdfn = 30.0
    
    obs = np.sort(obs)
    mod = np.sort(mod)
    
    global_max = max(np.amax(obs), np.amax(mod))
    
    wide = global_max / cdfn
    
    xbins = np.arange(0.0, global_max+wide, wide)
    
    pdfobs, bins = np.histogram(obs, bins=xbins)
    pdfmod, bins = np.histogram(mod, bins=xbins)
    
    cdfobs = np.insert(np.cumsum(pdfobs), 0, 0.0)
    cdfmod = np.insert(np.cumsum(pdfmod), 0, 0.0) 
    
    vals = [150., 256.6, 100000]
    
    def bias_map(vals, xbins, cdfmod, cdfobs):
        xbins = xbins
        cdfmod = cdfmod
        cdfobs = cdfobs
        
        cdf1 = np.interp(vals, xbins, cdfmod)
        
        corrected = np.interp(cdf1, cdfobs, xbins)
        
        return corrected 

    results = bias_map(vals, xbins, cdfmod, cdfobs)
        
    return results 

bias_corrected = rdd.map(bias_correct)

bias_corrected.take(10)

bias_corrected.first().mean()



