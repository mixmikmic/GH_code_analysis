get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import S3processing as s

prefix_eur = '/eodata/Sentinel-3/OLCI/OL_1_EFR/2017/05/21/S3A_OL_1_EFR____20170520T095410_20170520T095710_20170521T141215_0179_018_022_2159_LN1_O_NT_002.SEN3/'
#prefix_arc = '/eodata/Sentinel-3/OLCI/OL_1_EFR/2017/06/13/S3A_OL_1_EFR____20170613T192135_20170613T192239_20170613T214644_0064_018_370_1195_SVL_O_NR_002.SEN3/'

S3prod = s.OLCIprocessing(prefix_eur)
#S3prod = s.OLCIprocessing(prefix_arc)

#S3prod.ImportIMG()

S3prod.importNetCDF(NumBand=10)

S3prod.calcRGB(method='log')

S3prod.scaling(scale=8)

dmap = S3prod.createDynmap()
dmap

S3prod.transformCoords()

S3prod.createBasemap()

S3prod.savePNG(array=dmap.last.data)

S3prod.mapPlot(array=dmap.last.data)

S3prod.MercatorPlot()



