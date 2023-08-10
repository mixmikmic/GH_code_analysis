from IPython.display import Image
Image(filename='/home/andre/Pictures/autoseis.png', width=300)

ls

get_ipython().run_cell_magic('time', '', '\nimport glob\nimport os\n\n# address where the JD and relational files are\nrelationals = glob.glob(\'Linha_16*txt\') # all relational files with prefix Linha_16\n\nfor relational_file in relationals: # for every (station, relational) Julian Day file\n    f = open(relational_file,\'r\') # read the relation station and relational number\n    stations = f.readlines()\n    f.close()\n    #station, relational_rds =  stations[0].strip().split(\'\\t\') # split it\n    #print (station, relational_rds)    \n    JD = \'\'\n    try:        \n        JD = relational_file.split(\'.\')[0].split(\'_\')[-1]\n        os.chdir(glob.glob(\'*\'+JD)[0])    \n        os.chdir(\'SEGY\')    \n    except:\n        # continue to the next file\n        continue\n        \n    from obspy.segy import segy\n\n    f = open(\'..\\\\..\\\\JD_\'+ JD +\'_TR.txt\',\'w\')  # the trace header Report File \n    # every trace produces a pair\n    # station, pv\n    i=0\n    n = len(stations)\n\n    for station in stations: # station and station relational number (segy file)\n        try:\n            station, relational_rds =  station.strip().split(\'\\t\')\n            sgyfilename = glob.glob(\'*\'+relational_rds.strip()+\'*\')[0] # find the proper sgy file in this folder\n            print("file is ", sgyfilename, " : %.1f"%(100.0*i/n))          \n        except:\n            print ("Warn: couldn\'t find station relation segy ", relational_rds.strip())\n            i = i + 1\n            continue    \n        i = i + 1\n        try:\n            sgy = segy.readSEGY(sgyfilename, headonly=True, unpack_headers=True)  # just headers read    \n            pvbefore = 0\n            for trc in sgy.traces: # for every trace                \n                if trc.header.trace_value_measurement_unit != pvbefore: # avoid printing 3 sweeps, print just one\n                    pvbefore = trc.header.trace_value_measurement_unit\n                    f.write(station+\' \'+str(pvbefore)+\'\\n\')\n        except Exception as inst: \n            print (inst)\n            print ("Error: couldn\'t read relation segy ", relational_rds.strip())\n            continue    \n    f.close()    \n    os.chdir(\'..\') \n    os.chdir(\'..\') ')

# address where the TR files are
get_ipython().magic('cd C:\\Users\\alferreira.ANP.000\\Desktop\\HDs GLOBAL\\Linha 10')

# first create a concatenated TR file with all
import glob
import os
import pandas as pd
import pandas
import numpy as np

# Windows!
os.system("erase all_TR.txt")  # Linux os.system("rm all_TR.txt")
tr_files = glob.glob('*_TR.txt') # all trace report files each one for a JD
concat_tr_files = 'type ' # Linux 'cat '
for tr_file in tr_files:
    concat_tr_files += ' '+tr_file
concat_tr_files += '> JD_all_TR.txt' 
os.system(concat_tr_files)

tr_files = glob.glob('*_TR.txt') # all trace report files each one for a JD

for tr_file in tr_files: # for every trace report file create a PV_REPORT
    JD = tr_file.split('_TR')[0]
    st_pv = pandas.read_csv(tr_file, sep=' ', names=["station", "pv"])
    pv_st = st_pv.sort(columns="pv")
    pv_grouped = pv_st.groupby(by=["pv"])
    
    f = open(JD+'_PV_REPORT.txt','w') # the PV header report 

    for gp in pv_grouped: # for every pv group of stations
        pv = gp[0] # get the pv number
        stations = gp[1] # get all the stations
        first_st = stations.station.min() # first station
        last_st = stations.station.max() # last station
        # full range of recorded statations to see if is there any missing station
        range_rec = list(range(first_st, last_st+1)) 
        # how many missing based on acquisition geometry
        fst = pv - 280 # expected first station
        lst = pv + 280 # expected last station
        range_geometry = list(range(fst, lst+1))
        range_geometry.remove(pv) # split spread does not record on the vibration point
        # get how many stations recorded this pv on geometry range
        missing = 560 - abs(np.intersect1d(stations.station.values, pd.Series(range_geometry)).size)
        # get how many stations recorded this pv even if outside geometry    
        nrec = np.intersect1d(stations.station.values, pd.Series(range_rec)).size         
        f.write(str(pv)+' '+str(first_st)+' '+str(last_st)+' '+str(nrec)+' '+str(missing)+'\n')    
    f.close()    

#print (stations.station.values)
#print (pv + 280)
#print (list(range(fst, lst+1)[-3:]))
#print (missing)

get_ipython().magic('pylab inline')

import numpy
# concatenated version of all JD
pv_st_all = numpy.loadtxt('all_PV_REPORT.txt', delimiter=' ')

#***Checking if is there any fully missing pv's***
# PV range on the JD 84 to 78
#print pv_st_77_85R[:,0].min()
#print pv_st_77_85R[:,0].max()
#print arange(15613, 22957+2, 2)[-3:]
#print pv_st_77_85R[-3:,0]
#pv_st_77_85R[:,0].size - np.intersect1d(pv_st_77_85R[:,0], arange(15613, 22957, 2)).size

fig = plt.figure(figsize=(20,8), dpi=250)
ax = fig.add_subplot(111)
alive, = ax.plot(pv_st_all[:,0], pv_st_all[:,3], '.-b')
missing, = ax.plot(pv_st_all[:,0], pv_st_all[:,4], '.-r')
fullspread, = ax.plot(ax.get_xlim(), [560, 560], '--g')
halfspread, = ax.plot(ax.get_xlim(), [280, 280], '--g')
legend([alive, missing, fullspread, halfspread], ["Total Alive", "Missing Based On Geometry", "Full Spread", "Half Spread"], loc='upper left')

ax.set_ylabel('Number of Stations', fontsize=14)
ax.set_xlabel('PV', fontsize=14)
# get the range of PVs in each Julian Day
import numpy as np

pvr_files = glob.glob('*PV_REPORT.txt') # all trace report files each one for a JD
pvr_files.remove('JD_all_PV_REPORT.txt')

i = 0
y = np.array([0, 0])
for pvr_file in pvr_files:
    pv_st = numpy.loadtxt(pvr_file, delimiter=' ')
    labelPV = pvr_file.split('_PV_')[0]
    firstPV = pv_st[0,0]
    lastPV = pv_st[-1,0]       
    x = np.array([firstPV, lastPV])
    y += 100
    ax.plot(x, y, '-k')
    ax.plot(x, y, 'ok')
    ax.text(x.sum()/2, y[0]+20, labelPV.replace('_',' '), fontsize=14)
    i += 1
    
fig.savefig('PVReport.png', dpi=300)



