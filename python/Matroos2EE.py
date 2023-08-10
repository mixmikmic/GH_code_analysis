import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import pandas as pd
import requests
import io
import os
import datetime
from dateutil.relativedelta import relativedelta

PATH = '/tmp'

def readCSVforFusionTable():
    # Here keeping the source of Gerben's fnc.
    stations = ['Terschelling','Delfzijl','Haringvliet']

    for station in stations:
        print (station)
        df = pd.read_csv(r'D:\hagenaar\Documents\EMODNET\GLOSSISgrid\ObservedWaterlevel' + str(station) + '.txt', 
                         skiprows=13, delimiter='  ', names=['time','waterlevel'])

        df['time'] = pd.to_datetime(df['time'], yearfirst=True, format='%Y%m%d%H%M')
        df.set_index('time', inplace=True)
        df['waterlevel'][df['waterlevel'].isnull()] = np.nan
        df['waterlevel'][df['waterlevel'] == 9999999] # missing value
        df.dropna(inplace=True)
        df.to_csv(r'D:\hagenaar\Documents\EMODNET\GLOSSISgrid\ObservedWaterlevel' + str(station) + 'MODIFIED.txt')

listbuoys = {'terschelling noordzee': 297, 'harlingen': 6, 'nes': 321, 'delfzij': 1, 'haringvliet 10': 64}

t0 = datetime.datetime(2015, 1, 1)
tf = datetime.datetime(2018, 1, 1)

tnow = datetime.datetime.now().isoformat().replace('T','+')
tstart = t0.isoformat().replace('T','+') # start time
tend = tf.isoformat().replace('T','+') # end time

# function for subsequent lambda function
def df_get_values(value, default):
    if len(value.split())==2:
        v = value.split()[1]
    else:
        v = default
    return v

cc = {}

# This could take several minutes, make it year by year to improve performance.

for loc, loc_id0 in listbuoys.items():  # get water level signal for each buoy
    c_df_tot = pd.DataFrame() # empty db
    ti_dum = te_dum = t0
    while ti_dum<tf:
        te_dum = ti_dum + relativedelta(months=3) # add 3months
        
        # reconvert before request
        ti_dum_text = ti_dum.isoformat().replace('T','+')
        te_dum_text = te_dum.isoformat().replace('T','+')
        
        # in the following template,  loc_id0 is the location, source_newid0 is the "observed" type of source
        additional_params = "colors0=blue&localtime_offset=0&numser=1&old_unit_id0=1&oldlock_colors0=&oldlock_loc_id0=1&oldlock_source_newid0=1&oldlock_unit_id0=1&prealert=0&source_id0=10"
        url = ("http://matroos.deltares.nl/timeseries/start/series.php?"
                            "loc_id0=" + str(loc_id0) + "&" # this is the location of the buoy
                            "source_newid0=10&" # this is the source "observed"
                            "submit=Submit&"
                            "tcurrent=" + tnow + "&"
                            "tcurrent_new=" + tnow + "&"
                            "tstart=" + ti_dum_text + "&"
                            "tstop=" + te_dum_text + "&"
                            "type=noos&"
                            "unit_id0=1&"
                            "alarm=0&" # additional parameters from now on
                            + additional_params)

        r = requests.get(url).content
        c = pd.read_csv(io.StringIO(r.decode('utf-8')),sep='\t', header=0)
        c_head = c[0:11]
        # print(c_head[4:6]+'\n') # query location, if needed
        print(loc + ' - ' + 'from ' + ti_dum_text + ' to ' + te_dum_text)
        c_cont = c[12:].reset_index(drop=True) # reset index

        # get time and z from df
        c_t = c_cont.applymap(lambda x: datetime.datetime(
            int(x.split()[0][0:4]),
            int(x.split()[0][4:6]),
            int(x.split()[0][6:8]),
            int(x.split()[0][8:10]),
            int(x.split()[0][10:12])))
        c_t.columns = ['Time']

        c_z = c_cont.applymap(lambda x: df_get_values(x,np.NaN))
        c_z.columns = ['WaterLevel']

        c_df = c_t.join(c_z)
        c_df_tot = c_df_tot.append(c_df)
        
        ti_dum = te_dum
        
    # dictionary of Dataframes
    cc[loc] = c_df_tot # you can save it anywhere from here.


# save dict of df's into several files [per location], csv format.
for dfll_t, dfll_v in cc.items():
    dfll_v.to_csv(os.path.join(PATH,dfll_t.replace(' ','_'))+'.csv', sep=',', index=False, na_rep='NaN')
    

PATH = '/tmp'

bbox = {'waddenSea': [5.0048, 53.1737, 5.7354, 53.5115], 'westernScheldt': [3.2484, 51.3784, 3.9817, 51.7625]}

modelname = 'dcsmv6_zunov4_zuno_kf_hirlam'

t0 = datetime.datetime(2018, 1, 1)
tf = datetime.datetime(2018, 4, 1)

tnow = datetime.datetime.now().isoformat().replace('T','+')
tstart = t0.isoformat().replace('-','').replace('T','').replace(':','') # start time
tend = tf.isoformat().replace('-','').replace('T','').replace(':','') # end time

# This could take several minutes, make it year by year to improve performance.

for bboxloc, bboxloc_id0 in bbox.items():  # get water level signal for each buoy
    c_df_tot = pd.DataFrame() # empty db
    ti_dum = t0
    #while ti_dum<tf:
    te_dum = tf
        
    # reconvert before request
    ti_dum_text = ti_dum.isoformat().replace('T','+')
    te_dum_text = te_dum.isoformat().replace('T','+')

    # in the following template,  loc_id0 is the location, source_newid0 is the "observed" type of source
    additional_params = ""
    url = ("http://matroos.deltares.nl/direct/get_subgrid_ascii.php?"
                        "source=" + modelname + "&"  # this is the model name
                        "unit=SEP&" 
                        "tstart=" + tstart + "&"
                        "tstop=" + tend + "&"
                        "xmin=" + str(bboxloc_id0[0]) + "&"
                        "xmax=" + str(bboxloc_id0[2]) + "&"
                        "ymin=" + str(bboxloc_id0[1]) + "&"
                        "ymax=" + str(bboxloc_id0[3]) + "&"
                        "coordsys=wgs84"
                        + additional_params)
    
    r = requests.get(url).content
    c = pd.read_csv(io.StringIO(r.decode('utf-8')),sep='\t', header=0)
    #c_head = c[0:15]
    # print(c_head[4:6]+'\n') # query location, if needed
    print(bboxloc + ' - ' + 'from ' + ti_dum_text + ' to ' + te_dum_text)
    #c_cont = c[16:].reset_index(drop=True) # reset index

    # # get time and z from df
    #c_t = c_cont.applymap(lambda x: datetime.datetime(
    #    int(x.split()[0][0:4]),
    #    int(x.split()[0][4:6]),
    #    int(x.split()[0][6:8]),
    #    int(x.split()[0][8:10]),
    #    int(x.split()[0][10:12])))
    #c_t.columns = ['Time']

    #c_z = c_cont.applymap(lambda x: df_get_values(x,np.NaN))
    #c_z.columns = ['WaterLevel']

    #c_df = c_t.join(c_z)
    #c_df_tot = c_df_tot.append(c_df)

    # df of file - # you can save it anywhere from here.
    c.to_csv(os.path.join(PATH,bboxloc)+'.csv', sep=',', index=False, na_rep='NaN')

# save dict of df's into several files [per location], csv format.
for dfll_t, dfll_v in cc.items():
    dfll_v.to_csv(os.path.join(PATH,dfll_t.replace(' ','_'))+'.csv', sep=',', index=False, na_rep='NaN')
    

r



