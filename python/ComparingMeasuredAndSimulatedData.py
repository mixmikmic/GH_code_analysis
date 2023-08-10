get_ipython().run_line_magic('matplotlib', 'inline')

#Load all libraries
import pandas as pd
import datetime
from datetime import timedelta
import time
import brewer2mpl

# settings for graphics
# rcParams['figure.figsize'] = 20, 10
# rcParams['font.size'] = 16
bmap = brewer2mpl.get_map('Set3', 'qualitative', 12)
bmap2 = brewer2mpl.get_map('Set1', 'qualitative', 9)
colors = bmap.mpl_colors
colors2 = bmap2.mpl_colors
# rcParams['axes.color_cycle'] = colors2

# general settings
show_images = True # show equations etc.?
language_german = False; # False -> english, True -> german

HeatingSystemMeasurementData = pd.read_csv('MeasuredHeatingData2.csv',index_col='Date_Time', parse_dates=True, dayfirst=True)

HeatingSystemMeasurementData.resample('30min').mean().plot(figsize=(20,10));

HeatingSystemMeasurementData.head()

if language_german:
    ylabel_str = "Heizenergie [kWh] / Temperatur [C] / Durchflussrate [l/s]"
    xlabel_str = 'Datum'
    title_str  = "Messdaten Heizungssystem"
    label_str = ['Durchfluss','Vorlauftemperatur','Rücklauftemperatur','Heizenergie']
else:
    ylabel_str = "Heating Energy [kWh] / Temperature [C] / Flow Rate [l/s]"
    xlabel_str = 'Date'
    title_str = "Measured Heating System Data"
    label_str = ['Water Flow Rate','Supply Temperature','Return Temperature','Heating Energy']
    
ContractorHeatingMeasure = HeatingSystemMeasurementData.resample('30min').mean().plot(figsize=(25,10))
ContractorHeatingMeasure.set_ylabel(ylabel_str); ContractorHeatingMeasure.set_title(title_str); ContractorHeatingMeasure.set_xlabel(xlabel_str);
ContractorHeatingMeasure.legend(label_str,loc=4)
# plt.savefig('Measured_Data.pdf')

MeasuredHeatingData = pd.DataFrame(HeatingSystemMeasurementData['Warmefluss '].resample('H').mean())
MeasuredHeatingData.columns = ['Measured Data']
MeasuredHeatingData

MeasuredTempData = pd.read_csv('MeasuredTempData2.csv',sep=',', index_col='timestamp')

MeasuredTempData#.resample('D')

# get indices for columns that contain 'INTtemperature'
idx = [i for i, col in enumerate(MeasuredTempData.columns) if 'INTtemperature' in col]
#idx

InteriorTemperatures = MeasuredTempData[MeasuredTempData.columns[idx]]#.resample('30min').truncate(before='2013-01-30')
IntTemp = InteriorTemperatures.plot(figsize=(25,10),xticks=range(0,850,96))
if language_german:
    ylabel_str = "Raumtemperatur [C]"; xlabel_str = "Datum"; title_str = "Raumteperaturen Gemessen"
else:
    ylabel_str = "Zone Temperature [C]"; xlabel_str = "Date"; title_str = "Measured Indoor Temperatures"
    
IntTemp.set_ylabel(ylabel_str); IntTemp.set_xlabel(xlabel_str); IntTemp.set_title(title_str)

handles, labels = IntTemp.get_legend_handles_labels()
labels = [l.replace("('",'').replace("', 'INTtemperature')",'') for l in labels]
if language_german:
    labels = [l.replace('Buro','Büro').replace('Treppe','Treppenhaus') for l in labels]
else:
    labels = [l.replace('Buro','Office').replace('Treppe','Stairwell') for l in labels]

IntTemp.legend(labels)
labels = IntTemp.get_xticklabels();
labels = [l.get_text().replace(' 00:00:00','') for l in labels]
IntTemp.set_xticklabels(labels,rotation=30);
# plt.savefig('MeasuredIndoorTemperatures.pdf')

#Function to convert timestamps
def eplustimestamp(simdata,year_start_time=2013):
    timestampdict={}
    for i,row in simdata.T.iteritems():
        timestamp = str(year_start_time) + row['Date/Time']
        try:
            timestampdict[i] = datetime.datetime.strptime(timestamp,'%Y %m/%d  %H:%M:%S')
        except ValueError:
            tempts = timestamp.replace(' 24', ' 23')
            timestampdict[i] = datetime.datetime.strptime(tempts,'%Y %m/%d  %H:%M:%S')
            timestampdict[i] += timedelta(hours=1)
    timestampseries = pd.Series(timestampdict)
    return timestampseries

def loadsimdata(file,pointname,ConvFactor,year_start_time=2013):
    df = pd.read_csv(file)
    df['desiredpoint'] = df[pointname]*ConvFactor
    df.index = eplustimestamp(df,year_start_time)
    pointdf = df['desiredpoint']
    return pointdf

Simlist = ['Sim1Data.csv','Sim2Data.csv','Sim3Data.csv','Sim4Data.csv']
SimHeatingDataList = []
for file in Simlist:
    print 'Loading '+file
    x = loadsimdata(file,'EMS:All Zones Total Heating Energy {J}(Hourly)',0.0000002778)
    SimHeatingDataList.append(x)

SimHeatingData = pd.concat(SimHeatingDataList, axis=1, keys=Simlist)

SimHeatingData.resample('D').mean().plot(figsize=(20,10))

CombinedHeating = pd.merge(SimHeatingData, MeasuredHeatingData, right_index=True, left_index=True)
CombinedHeating.head()

do_truncate = True;

if do_truncate:
    SimVsMeasHeating = CombinedHeating.truncate(after='2013-02-06').plot(figsize=(25,10),linewidth=2)
else:
    SimVsMeasHeating = CombinedHeating.plot(figsize=(25,10),linewidth=2)

if language_german:
    ylabel_str = 'Heizenergie [kWh]'; xlabel_str = "Datum"; title_str = 'Vergleich Messung / Simulation';
    labels = ['Simulation, Iteration 1', 'Simulation, Iteration 2', 'Simulation, Iteration 3', 'Simulation, Iteration 4', 'Messdaten']
else:
    ylabel_str = 'Heating Energy [kWh]'; xlabel_str = "Date"; title_str = 'Measured vs. Simulated Heating Comparison';
    labels = ['Simulation, Iteration 1', 'Simulation, Iteration 2', 'Simulation, Iteration 3', 'Simulation, Iteration 4', 'Measured Data']

SimVsMeasHeating.set_ylabel(ylabel_str); SimVsMeasHeating.set_xlabel(xlabel_str); SimVsMeasHeating.set_title(title_str);
SimVsMeasHeating.legend(labels,loc=4)

# if do_truncate:
#     savefig('Measured_vs_Simulated_zoom.pdf')
# else:
#     savefig('Measured_vs_Simulated.pdf')

from __future__ import division



CombinedHeating

dataset = 'Sim4Data.csv'
NMBE = 100*(sum(CombinedHeating['Measured Data'] - CombinedHeating[dataset] )/(CombinedHeating['Measured Data'].count()*CombinedHeating['Measured Data'].mean()))
CVRSME = 100*((sum((CombinedHeating['Measured Data'] - CombinedHeating[dataset] )**2)/(CombinedHeating['Measured Data'].count()-1))**(0.5))/CombinedHeating['Measured Data'].mean()

print 'NMBE: ' + str(round(NMBE,2)) + '    CVRSME : ' + str(round(CVRSME,2))

from IPython.core.display import Image
Image(filename='./ashrae14calibrationmetrics.png')

SimRetrofitList = ['Sim4Data.csv',
                   'Retrofit1_Windows.csv',
                   'Retrofit1_Plaster.csv',
                   'Retrofit2_Aerogel.csv',
                   'Retrofit1_Ceiling.csv',
                   'Retrofit1_AirtightnessHigh.csv',
                   'Retrofit1.csv',
                   'Retrofit2.csv']
SimRetrofitDataList = []
for file in SimRetrofitList:
    try:
        x = loadsimdata('./'+file,'EMS:All Zones Total Heating Energy {J}(Hourly)',0.0000002778,"2012")
    except: 
        continue
    SimRetrofitDataList.append(x)

def get_retrofit_labels(labels,language_german):
    if language_german:
        labels = [l.replace('Sim4HC_SB3_','Ausgangszustand + ').replace('Sim4HC_SB3','Ausgangszustand').replace('1_',' nur ').replace('2_',' nur ').replace('Annual','').replace('Ausgangszustand + Retrofit','Retrofit ') for l in labels]
    else:
        labels = [l.replace('Sim4HC_SB3_','Baseline + ').replace('Sim4HC_SB3','Baseline').replace('1_',' only ').replace('2_',' only ').replace('Annual','').replace('Baseline + Retrofit','Retrofit') for l in labels]
    return labels

SimRetrofitData = pd.concat(SimRetrofitDataList, axis=1, keys=SimRetrofitList)
SimRetrofitHeating = SimRetrofitData.tshift(-1,freq='H').resample('D').sum().plot(figsize=(25,10),linewidth=1)

handles, labels = SimRetrofitHeating.get_legend_handles_labels()

if language_german:
    ylabel_str = 'Gesamt-Heizenergie pro Tag [kWh]'; xlabel_str = "Datum"; title_str = 'Vergleich Simulation Ausgangszustand und Renovations-Szenarien';
else:
    ylabel_str = 'Total Heating Energy per Day [kWh]'; xlabel_str = 'Date'; title_str = 'Simulated Baseline vs. Retrofit Scenarios Comparison';

SimRetrofitHeating.set_ylabel(ylabel_str); SimRetrofitHeating.set_xlabel(xlabel_str); SimRetrofitHeating.set_title(title_str)

SimRetrofitData = pd.concat(SimRetrofitDataList, axis=1, keys=SimRetrofitList);
SimRetrofitDataHeatingMonthly = SimRetrofitData.tshift(-1,freq='H').resample('M').sum().plot(figsize=(25,10),kind='bar');

handles, labels = SimRetrofitDataHeatingMonthly.get_legend_handles_labels();

if language_german:
    ylabel_str = 'Gesamt-Heizenergie pro Monat [kWh]'; xlabel_str = 'Monat der Simulationsperiode'; title_str = 'Vergleich Simulation Ausgangszustand und Renovations-Szenarien';
else:
    ylabel_str = 'Total Heating Energy per Month [kWh]'; xlabel_str = 'Month of Simulation Period'; title_str = 'Simulated Baseline vs. Retrofit Scenarios Comparison';

SimRetrofitDataHeatingMonthly.set_ylabel(ylabel_str); SimRetrofitDataHeatingMonthly.set_xlabel(xlabel_str); SimRetrofitDataHeatingMonthly.set_title(title_str);

SimRetrofitDataHeatingMonthly.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],rotation=0);

def get_air_temperatures_of_conditioned_zones(filename,unconditioned_zones):
    data = pd.read_csv(filename)
    data.index = eplustimestamp(data)
    columnlist = pd.Series(data.columns)
    columnlist = list(columnlist[(columnlist.str.endswith("Zone Mean Air Temperature [C](Hourly)"))])
    for zonename in unconditioned_zones: # filter out unconditioned zones
        columnlist = filter(lambda columnname: not zonename in columnname,columnlist)
    return data[columnlist]


def get_number_of_hours_not_comfortable(filename,unconditioned_zones):
    # settings
    beginocc = 6; endocc = 23; # hours occupied: beginocc < x < endocc
    endheating = 6; beginheating = 8; # months of heating period: x < endheating OR x > beginheating
    tempthreshold = 19.5
    
    # get data
    data = get_air_temperatures_of_conditioned_zones(filename,unconditioned_zones)
    
    # count uncomfortable hours
    d = dict()
    for rowname in data: 
        row = data[rowname]
        d[rowname.split(':')[0]] = len(row[(row < tempthreshold) 
                           & (row.index.hour > 6) & (row.index.hour < 23)
                           & ((row.index.month > beginheating) | (row.index.month < endheating))  ])
    return d, sum(d.values())

filename = 'Sim4Data.csv'

unconditioned_zones = ['ZONE_U1_W', 'ZONE_U1_N', 'ZONE_U1_ST', 'ZONE_00_ST', 'ZONE_01_ST', 'ZONE_02_ST', 
                       'ZONE_03_ST', 'ZONE_04_ST', 'ZONE_04_N', 'ZONE_05_N', 'ZONE_05_S']
unconditioned_zones.append('ZONE_U1_LA') # many uncomfortable hours here...

total_hours_not_comfortable = dict()
for filename in SimRetrofitList:
    try:
        hours_not_comfortable, N = get_number_of_hours_not_comfortable("./"+filename,unconditioned_zones)
        #print hours_not_comfortable # print per zone
        total_hours_not_comfortable[filename] = int(N/39.)
        print filename, int(N/39.)   # total, normalized by number of zones
    except:
        continue

#fig = figure(figsize=(8,6),dpi=300, facecolor='w', edgecolor='k');
ComfortData = pd.Series(total_hours_not_comfortable)
ComfortData
ComfortPlot = ComfortData.plot(kind='bar')
ComfortPlot.set_title('Avg. Hours Uncomfortable')



