from IPython.display import Image
Image("Figures/EbbTideCurrent.jpg")

Image("Figures/SlackTide.jpg")

Image("Figures/FloodTideCurrent.jpg")

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed
from tydal.module3_utils import plot_currents
get_ipython().magic('matplotlib inline')

interact(plot_currents,T=fixed(12.42),a1=[0,4],a2=[0,4],alpha=(0,90),N=(0,399))

import tydal.module3_utils as m3
import tydal.module2_utils as tu

URL1='http://107.170.217.21:8080/thredds/dodsC/Salish_L1_STA/Salish_L1_STA.ncml'

[ferry, ferry_download, message]=m3.ferry_data_download(URL1)

ferryQC= m3.ferry_data_QC(ferry,6.5,4,4)

ferryQC = m3.count_route_num(ferryQC[0])

#import tides
pt_tide = tu.load_Port_Townsend('Data/')
pt_tide = pt_tide['Water Level']

start_date = '2016-10-01'
end_date = '2016-11-01'

#plt.style.use('ggplot')
get_ipython().magic('matplotlib inline')
interact(m3.plt_ferry_and_tide, ferryQc=fixed(ferryQC), 
         pt_tide=fixed(pt_tide), crossing_index = (0,280),
        start_date = fixed(start_date), end_date = fixed(end_date))

import tydal.quiz3

tydal.quiz3



