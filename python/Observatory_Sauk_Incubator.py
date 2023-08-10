#Python libraries available on CUAHSI JupyterHub 
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')

#HydroShare Utilities
from utilities import hydroshare

hs=hydroshare.hydroshare()

get_ipython().system('wget https://www.hydroshare.org/django_irods/download/2474e3c1f33b4dc58e0dfc0824c72a84/data/contents/ogh_meta.json')
get_ipython().system('wget https://www.hydroshare.org/django_irods/download/2474e3c1f33b4dc58e0dfc0824c72a84/data/contents/ogh.py')

import ogh
homedir = ogh.mapContentFolder(str(os.environ["HS_RES_ID"]))
print('Data will be loaded from and save to:'+homedir)

hs.getResourceFromHydroShare('c532e0578e974201a0bc40a37ef2d284')
shapefile = hs.content['wbdhuc12_17110006_WGS84.shp']

hs.getResourceFromHydroShare('ef2d82bf960144b4bfb1bae6242bcc7f')
NAmer = hs.content['NAmer_dem_list.shp']

mappingfile = ogh.treatgeoself(shapefile=shapefile, NAmer=NAmer, folder_path=os.getcwd(), outfilename='monkeysonatree.csv', buffer_distance=0.00)
print(mappingfile)

loc_name='Skagit Watershed'
streamflow_watershed_drainage_area=8010833000 # square meters

#Assuming this is pulled from Github, how can we import this from Utilities.
#Otherwise it needs to be in each HydroShare resource - which if fine too. 
with open('ogh_meta.json','r') as r:
    meta_file = json.load(r)
    r.close()

sorted(meta_file.keys())

hs.getResourceFromHydroShare('abe2fd1e3fc74e889b78c2701872bd58')

get_ipython().system('cd /home/jovyan/work/notebooks/data/abe2fd1e3fc74e889b78c2701872bd58/abe2fd1e3fc74e889b78c2701872bd58/data/contents')
get_ipython().system('pwd')



print('This is the list of folders in your directory for this HydroShare resource.')
test = [each for each in os.listdir(homedir) if os.path.isdir(each)]
print(test)

ThisNotebook='Observatory_Skagit_LivBC2WRFlow_122117.ipynb' #check name for consistency

liv2013_tar = 'livneh2013.tar.gz'
wrf_tar = 'salathe2014.tar.gz'
biascorrWRF_liv_tar = 'biascorrWRF_liv.tar.gz'
biascorrWRF_global_tar = 'biascorrWRF_global.tar.gz'

get_ipython().system('tar -zcf {liv2013_tar} livneh2013')
get_ipython().system('tar -zcf {wrf_tar} salathe2014')
get_ipython().system('tar -zcf {biascorrWRF_liv_tar} biascorrWRF_liv')
get_ipython().system('tar -zcf {biascorrWRF_global_tar} biascorrWRF_global')

observatory_gridded_hydromet='ogh.py'
soil = 'soil'
CorrectionFactors_wrfliv='BiasCorr_wrfbc.json'
CorrectionFactors_lowliv='BiasCorr_wrfbc_lowLiv.json'
listofgridpoints ='monkeysonatree.csv'

files=[ThisNotebook,
       liv2013_tar,
       observatory_gridded_hydromet,
       wrf_tar,
       biascorrWRF_liv_tar,biascorrWRF_global_tar,
       soil,listofgridpoints,
       CorrectionFactors_wrfliv,CorrectionFactors_lowliv]

# for each file downloaded onto the server folder, move to a new HydroShare Generic Resource
title = 'Skagit Observatory Bias Correction Results - Livneh et al., 2013 to WRF (Salathe et al., 2014) and low elevation spatial average correction.'
abstract = 'This output is a bias correction test to generate a hybrid gridded meteorology product. This dataset was generated December 21, 2017 using Observatory code from https://github.com/ChristinaB/Observatory.'
keywords = ['Sauk', 'climate', 'WRF','hydrometeorology'] 
rtype = 'genericresource'  

# create the new resource
resource_id = hs.createHydroShareResource(abstract, 
                                          title,
                                          keywords=keywords, 
                                          resource_type=rtype, 
                                          content_files=files, 
                                          public=False)

#check name for consistency
get_ipython().system(' cp Observatory_Sauk_Incubator.ipynb /home/jovyan/work/notebooks/data/Incubating-a-DREAM/Sauk_JupyterNotebooks')

get_ipython().system('ls')



