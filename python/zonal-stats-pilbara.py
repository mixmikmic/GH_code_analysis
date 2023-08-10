import numpy
import fiona
import pandas
import geopandas
from image_processing import contrast
from image_processing.segmentation import Segments, rasterise_vector
import matplotlib.pyplot as plt
import datacube
from datacube.storage import masking
dc = datacube.Datacube(app='dc-example')
get_ipython().magic('matplotlib inline')

vfname = '/g/data/v10/testing_ground/jps547/pilbara/CaneRiver_Nanutarra_MtMinnie_Peedamulla_RedHill_Yarraloola_Merge_WGS84_LandSystemType.shp'
src = fiona.open(vfname, 'r')
xidx = (src.bounds[0], src.bounds[2])
yidx = (src.bounds[-1], src.bounds[1])
gdf = geopandas.read_file(vfname)
gdf.plot()

print xidx
print yidx
xidx = (115.1, 115.9)
yidx = (-22.1, -22.5)
print xidx
print yidx

nbar_ls5 = dc.load(product='ls5_nbar_albers', x=xidx, y=yidx, time=('2005-04', '2006-04'), group_by='solar_day')

pq_ls5 = dc.load(product='ls5_pq_albers', x=xidx, y=yidx, time=('2005-04', '2006-04'), group_by='solar_day')

nbar_ls5 = nbar_ls5.sel(time=pq_ls5.time)

nbar_ls5.nbytes / 1024.0 / 1024.0 / 1024.0

mask = masking.make_mask(pq_ls5, ga_good_pixel=True)

nbar_ls5 = nbar_ls5.where(mask.pixelquality)

ha = nbar_ls5.affine.a **2 / 10000.0
print "Pixel area in hectares: {}".format(ha)

dims = nbar_ls5.dims

def get_rgb(dataset, bands, time, window=None, percent=2):
    if window is None:
        dset_subs = dataset[bands].sel(time=time)
    else:
        xs, ys = (window[1][0], window[0][0]) * dataset.affine
        xe, ye = (window[1][1], window[0][1]) * dataset.affine
        dset_subs = dataset[bands].sel(time=time, x=slice(xs, xe), y=slice(ys, ye))
    dims = dset_subs.dims
    rgb = numpy.zeros((dims['y'], dims['x'], 3), dtype='uint8')
    for i, band in enumerate(bands):
        rgb[:, :, i] = contrast.linear_percent(dset_subs[band].values, percent=percent)
    return rgb

nbar_ls5.time.values

rgb = get_rgb(nbar_ls5, ['swir1', 'red', 'green'], '2006-02-19', percent=5)
plt.imshow(rgb)

ras = rasterise_vector(vfname, shape=(dims['y'], dims['x']), crs=nbar_ls5.crs.wkt, transform=nbar_ls5.affine)
seg = Segments(ras)
print "Number of segments: {}".format(seg.n_segments)

spectra = nbar_ls5.data_vars.keys()
zonal_stats = pandas.DataFrame()
for ts in nbar_ls5.time:
    spectra_stats = pandas.DataFrame(columns=['Segment_IDs'])
    for sp in spectra:
        data = nbar_ls5[sp].sel(time=ts).values
        df = seg.basic_statistics(data, nan=True, scale_factor=ha, dataframe=True, label=sp)
        spectra_stats = pandas.merge(spectra_stats, df, on='Segment_IDs', how='outer')
    spectra_stats['timestamp'] = ts.values
    zonal_stats = zonal_stats.append(spectra_stats)
zonal_stats.set_index('timestamp', inplace=True)

seg.ids

gdf.iloc[2]

zonal_stats.head(4)

df = zonal_stats[zonal_stats['Segment_IDs'] == 2]

mean_cols = [col for col in df.columns if 'Mean' in col]
max_cols = [col for col in df.columns if 'Max' in col]
min_cols = [col for col in df.columns if 'Min' in col]
stdv_cols = [col for col in df.columns if 'StdDev' in col]

df[stdv_cols].resample('M').mean().plot(title='Standard Deviation').legend(loc='center left', bbox_to_anchor=(1, 0.5))
df[mean_cols].resample('M').mean().plot(title='Mean').legend(loc='center left', bbox_to_anchor=(1, 0.5))
df[min_cols].resample('M').mean().plot(title='Min').legend(loc='center left', bbox_to_anchor=(1, 0.5))
df[max_cols].resample('M').mean().plot(title='Max').legend(loc='center left', bbox_to_anchor=(1, 0.5))

bboxes = seg.bounding_box()

nbar_ls5.time.values

bbox = bboxes[2]
rgb = get_rgb(nbar_ls5, ['nir', 'red', 'green'], '2005-07-01', bbox, 2)
plt.imshow(rgb, interpolation="nearest")

rgb = get_rgb(nbar_ls5, ['swir1', 'nir', 'red'], '2006-03-07', bbox, 2)
plt.imshow(rgb, interpolation='nearest')

store = pandas.HDFStore('pilbara-time-series-zonal-stats.h5', 'w', complib='blosc')
store['stats'] = zonal_stats

store.close()



