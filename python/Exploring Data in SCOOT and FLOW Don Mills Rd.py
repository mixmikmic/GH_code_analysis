get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
from matplotlib import cm
import pandas as pd
from pg import DB
import configparser
import numpy as np
import preprocess
import plotting_functions
from collections import OrderedDict
from sklearn import linear_model

CONFIG = configparser.ConfigParser()
CONFIG.read('db.cfg')
dbset = CONFIG['DBSETTINGS']
db = DB(dbname=dbset['database'],host=dbset['host'],user=dbset['user'],passwd=dbset['password'])

# Don Mills (Eglinton to Lawrence) SCN#
corridor = [70211, 12881, 12871, 12861, 12851, 12821]
corridor_tcl = [str(x) for x in [444715,444647,444376,4709500,10180209,20234118,30018925,4709528]]
seg_names = ['The Donway - Lawrence','Lawrence - Clock Tower Rd', 'Clock Tower Rd - The Donway','The Donway - Green Belt Dr','Green Belt Dr - Wynford Dr','Green Belt Dr - Wynford Dr','Wynford Dr - Eglinton Ave','Eglinton Ave - Rochefort Dr']
corridor_name = 'Don Mills Rd'

# Get list of all detectors
sdetectors = pd.DataFrame(db.query('SELECT scn,px,det,direction,sideofint FROM scoot.scoot_detectors').getresult(),columns=['scn','px','det','direction','sideofint'])
# Filter out detectors not along the specified corridor
sdetectors = sdetectors[sdetectors['scn'].isin(corridor)]
# Filter out turning movement detectors
sdetectors = sdetectors[sdetectors.direction.str.contains("LT")==False]
# Get mapping of the detectors 
sdet_tcl = pd.DataFrame(db.query('SELECT detector, centreline_id FROM scoot.detector_tcl').getresult(),columns=['det','centreline_id'])
# Merge mapping to detectors dataframe
sdetectors = sdetectors.merge(sdet_tcl, how='inner')
# Reformat detector names to match those of raw input tables
sdetectors['det'] = ['N'+x for x in list(sdetectors['det'])]
# Filter out legs that are perpendicular to the corridor
sdetectors = sdetectors[(sdetectors['direction']=='NB') | (sdetectors['direction']=='SB')]
# Filter out individual detectors (in this case, because they are measuring fleet st)
#sdetectors = sdetectors[(sdetectors['det']!='N30421X1') & (sdetectors['det']!='N30421Z1') & (sdetectors['det']!='N30331B1') & (sdetectors['det']!='N30411H1') & (sdetectors['det']!='N30431T1')]

start_year = 2011
start_month = 6
end_year = 2011    
end_month = 6
end_day = 30
gs = [(444647,'NB'),(10180209,'SB')]
colorss = ['b','y','g','k']
fdata = pd.DataFrame(db.query('SELECT centreline_id, count_bin, volume, dir_bin FROM prj_volume.centreline_volumes         WHERE centreline_id in (' + ','.join(corridor_tcl) + ') AND count_bin >= \'' + str(2010) + '-' +         str(1) + '-01\' AND count_bin <= \'' + str(2017) + '-' + str(12) + '-' + str(31) + '\'').getresult(), 
        columns = ['centreline_id','count_bin','volume','dir_bin'])
fdata = preprocess.preprocess_flow(fdata,'NS')
gf = [(444647,+1),(10180209,-1)]
colorsf = ['r','p','o','c']

for (g1,g2) in zip(gs,gf):
    fig,(ax1,ax2) = plt.subplots(1,2,figsize = (12,5))
    plt.title('Time of Day Profile for ' + corridor_name + ' ' + seg_names[corridor_tcl.index(str(g1[0]))] + ' '+ str(g1[1]) + ' from ' + str(start_year*100+start_month) + ' to '+ str(end_year*100+end_month),ha='right',va='baseline')
    for year in range(start_year, end_year+1):
        (sm,em) = preprocess.get_start_end_month(start_year, end_year, start_month, end_month, year)
        for month in range(sm, em):
            if month < 10:
                m = '0' + str(month)
            else:
                m = str(month)
            # Read in Raw Data
            #sdata = pd.DataFrame(db.query('SELECT detector, start_time, end_time, flow_mean FROM scoot.raw_'+str(year)+m).getresult(),columns=['detector', 'start_time', 'end_time', 'flow_mean'])        
            sdata = pd.read_table('DETS_'+str(year)+m+'.txt', delim_whitespace=True, skiprows=9, header = None, 
                                  names=['Site','DOW','Date','Time_Start','Time_End','flow_mean_veh/h'], usecols=range(6))
            sdata = preprocess.preprocess_scoot(sdata,sdetectors)

            # Plot TOD
            sncounts, savg, sndays = plotting_functions.TOD(ax1,sdata,'centreline_id','direction','Date','flow_mean_veh/h',g1,colorss,True,'SCOOT')
            sncounts, savg, sndays = plotting_functions.TOD(ax2,sdata,'centreline_id','direction','Date','flow_mean_veh/h',g1,colorss,False,'SCOOT')
            
    fncounts, favg, fndays = plotting_functions.TOD(ax1,fdata,'centreline_id','dir_bin','date','volume',g2,colorsf,True,'FLOW')
    fncounts, favg, fndays = plotting_functions.TOD(ax2,fdata,'centreline_id','dir_bin','date','volume',g2,colorsf,False,'FLOW')
        
    plt.legend()
    plt.show()

start_year = 2011
start_month = 7
end_year = 2011  
end_month = 7
end_day = 30

fig, (ax1, ax2) = plt.subplots(2, figsize=(10,7), sharex = True, sharey = True)

# SCOOT 
colorss = ['b','y','g','k']
print('Plotting Average Daily Volume from ' + str(start_year*100+start_month) + ' to '+ str(end_year*100+end_month) + '...')
for year in range(start_year, end_year+1):
    (sm,em) = preprocess.get_start_end_month(start_year, end_year, start_month, end_month, year)
    for month in range(sm, em):
        if month < 10:
            m = '0' + str(month)
        else:
            m = str(month)
        # Read in Raw Data
        #sdata = pd.DataFrame(db.query('SELECT detector, start_time, end_time, flow_mean FROM scoot.raw_'+str(year)+m).getresult(),columns=['detector', 'start_time', 'end_time', 'flow_mean'])        
        sdata = pd.read_table('DETS_'+str(year)+m+'.txt', delim_whitespace=True, skiprows=9, header = None, 
                              names=['Site','DOW','Date','Time_Start','Time_End','flow_mean_veh/h'], usecols=range(6))
        sdata = preprocess.preprocess_scoot(sdata,sdetectors)
        
        plotting_functions.daily_vol(ax1,ax2,sdata,'stopindex','direction','Date','flow_mean_veh/h',colorss,'SCOOT',corridor_name)

# FLOW
fdata = pd.DataFrame(db.query('SELECT centreline_id, count_bin, volume, dir_bin FROM prj_volume.centreline_volumes         WHERE centreline_id in (' + ','.join(corridor_tcl) + ') AND count_bin >= \'' + str(2010) + '-' +         str(1) + '-01\' AND count_bin <= \'' + str(2017) + '-' + str(12) + '-' + str(31) + '\'').getresult(), 
        columns = ['centreline_id','count_bin','volume','dir_bin'])
fdata = preprocess.preprocess_flow(fdata,'NS')
colorsf = ['r','p','o','c']
plotting_functions.daily_vol(ax1,ax2,fdata,'stopindex','dir_bin','date','volume',colorsf,'FLOW', corridor_name)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()

start_year = 2011
start_month = 1
end_year = 2011   
end_month = 12
end_day = 31
cm_subsection = np.append(np.linspace(0,0.8,6) ,np.linspace(0.9,0,6))
gs = [(444647,'NB'),(10180209,'SB')]
gf = [(1147466,-1),(1147201,-1),(1147201,+1)]

for g1, g2 in zip(gs, gf):
    fig,(ax1) = plt.subplots(1,1,figsize = (15,5))
    ax1.set_title('SCOOT')
    fig.suptitle(corridor_name + ' ' + seg_names[corridor_tcl.index(str(g1[0]))] + ' '+ str(g1[1]) + ' from ' + str(start_year*100+start_month) + ' to '+ str(end_year*100+end_month),fontsize=15)
    for year in range(start_year, end_year+1):
        (sm,em) = preprocess.get_start_end_month(start_year, end_year, start_month, end_month, year)
        for month in range(sm, em):
            if month < 10:
                m = '0' + str(month)
            else:
                m = str(month)
            # Read in Raw Data
            sdata = pd.read_table('DETS_'+str(year)+m+'.txt', delim_whitespace=True, skiprows=9, header = None, 
                                  names=['Site','DOW','Date','Time_Start','Time_End','flow_mean_veh/h','Occ_mean_%'], usecols=range(7))
            sdata = preprocess.preprocess_scoot(sdata,sdetectors)
            # Plot Seasonality
            plotting_functions.seasonality_plot(ax1,sdata,'centreline_id','direction','Date','flow_mean_veh/h',g1,'b')
    fig.autofmt_xdate()
    plt.show()

start_year = 2010
start_month = 9
end_year = 2016    
end_month = 12
end_day = 31

cm_subsection = np.linspace(0,1,12)
colors = [cm.jet(x) for x in cm_subsection]
gs = [(10180209,'SB'),(10180209,'NB')]
gf = [(10180209,-1),(10180209,+1)]

for g1, g2 in zip(gs, gf):  
    fig,(ax1) = plt.subplots(1,1,figsize=(5,5))
    # FLOW
    fdata = pd.DataFrame(db.query('SELECT centreline_id, count_bin, volume, dir_bin FROM prj_volume.centreline_volumes             WHERE centreline_id = ' + str(g2[0]) + ' AND count_bin >= \'' + str(start_year) + '-' +             str(start_month) + '-01\' AND count_bin <= \'' + str(end_year) + '-' + str(end_month) + '-' + str(end_day) + '\'             AND dir_bin = ' + str(g2[1])).getresult(), 
            columns = ['centreline_id','count_bin','volume','dir_bin'])
    fdata = preprocess.preprocess_flow(fdata,'NS')
    # Sum if volume is stored on a lane by lane basis
    fdata = fdata.groupby(['month', 'time_15', 'date', 'centreline_id', 'stopindex', 'dir_bin', 'direction'], as_index=False).sum()
    regr = linear_model.LinearRegression()
    t = fdata['date'].drop_duplicates()
    tt = []
    for item in t:
        if (item.year, item.month) not in tt:
            tt.append((item.year, item.month))

    # SCOOT
    for (year,month) in tt:
        # Read in Raw Data
        sdata = pd.read_table('DETS_'+str(year*100+month)+'.txt', delim_whitespace=True, skiprows=9, header = None, 
                              names=['Site','DOW','Date','Time_Start','Time_End','flow_mean_veh/h','Occ_mean_%'], usecols=range(7))
        sdata = preprocess.preprocess_scoot(sdata,sdetectors)
        sdata = sdata.groupby(['Date','centreline_id','stopindex','month','time_15','direction'], as_index=False).sum()

        data = pd.merge(sdata, fdata, how = 'inner', left_on = ['Date','time_15','stopindex','direction'], right_on = ['date','time_15','stopindex','direction'])
        data = data[(data['volume']!=0) & (data['flow_mean_veh/h']!=0)]
        
        ax1.scatter(data['volume'], data['flow_mean_veh/h'], color = colors[month-1], label=None)

    # Add y=x
    lims = [
    np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
    np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
    ]
    # plot both limits against eachother
    ax1.plot(lims, lims, 'k-', alpha=0.75, label = 'y=x')
    ax1.legend(loc=4,fontsize=10)
    plt.title(corridor_name + ' ' + seg_names[corridor_tcl.index(str(g1[0]))] + ' '+ str(g1[1]) + ' from ' + str(start_year*100+start_month) + ' to '+ str(end_year*100+end_month),fontsize=14)
    ax1.set_xlabel('Volume in FLOW (veh)')
    ax1.set_ylabel('Volume in SCOOT (veh)')
    plt.show()

db.close()



