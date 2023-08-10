import pandas as pd
import datetime
from datetime import timedelta
import time

get_ipython().run_line_magic('matplotlib', 'inline')

def loadsimdata(file,pointname,ConvFactor):
    df = pd.read_csv(file)
    df['desiredpoint'] = df[pointname]*ConvFactor
    df.index = eplustimestamp(df)
    pointdf = df['desiredpoint']
    return pointdf

SimulationData = pd.read_csv('EnergyplusSimulationData.csv')

SimulationData

SimulationData.info()

SimulationData['AIRNODE_ZONENODE_U1_S:System Node Setpoint Temp[C](Hourly) ']#.plot()

SimulationData['Date/Time'].tail()

#Function to convert timestamps
def eplustimestamp(simdata):
    timestampdict={}
    for i,row in simdata.T.iteritems():
        timestamp = str(2013) + row['Date/Time']
        try:
            timestampdict[i] = datetime.datetime.strptime(timestamp,'%Y %m/%d  %H:%M:%S')
        except ValueError:
            tempts = timestamp.replace(' 24', ' 23')
            timestampdict[i] = datetime.datetime.strptime(tempts,'%Y %m/%d  %H:%M:%S')
            timestampdict[i] += timedelta(hours=1)
    timestampseries = pd.Series(timestampdict)
    return timestampseries

SimulationData.index = eplustimestamp(SimulationData)

SimulationData.info()

SimulationData['AIRNODE_ZONENODE_U1_S:System Node Setpoint Temp[C](Hourly) ']

ColumnsList = pd.Series(SimulationData.columns)
ColumnsList.head(100)

ColumnsList.str.endswith("Zone Mean Air Temperature [C](Hourly)")

ColumnsList[(ColumnsList.str.endswith("Zone Mean Air Temperature [C](Hourly)"))]

ZoneTempPointList = list(ColumnsList[(ColumnsList.str.endswith("Zone Mean Air Temperature [C](Hourly)"))])
ZoneTempPointList

BasementZoneTemp = list(ColumnsList[(ColumnsList.str.endswith("Zone Mean Air Temperature [C](Hourly)"))&(ColumnsList.str.contains("U1"))])
GroundFloorZoneTemp = list(ColumnsList[(ColumnsList.str.endswith("Zone Mean Air Temperature [C](Hourly)"))&(ColumnsList.str.contains("00"))])
Floor1ZoneTemp = list(ColumnsList[(ColumnsList.str.endswith("Zone Mean Air Temperature [C](Hourly)"))&(ColumnsList.str.contains("01"))])
Floor2ZoneTemp = list(ColumnsList[(ColumnsList.str.endswith("Zone Mean Air Temperature [C](Hourly)"))&(ColumnsList.str.contains("02"))])
Floor3ZoneTemp = list(ColumnsList[(ColumnsList.str.endswith("Zone Mean Air Temperature [C](Hourly)"))&(ColumnsList.str.contains("03"))])
Floor4ZoneTemp = list(ColumnsList[(ColumnsList.str.endswith("Zone Mean Air Temperature [C](Hourly)"))&(ColumnsList.str.contains("04"))])

ZoneTemp = SimulationData[ZoneTempPointList]#.drop(['EMS:All Zones Total Heating Energy {J}(Hourly)'],axis=1)

ZoneTemp.info()

ZoneTemp.plot(figsize=(25,15))

ZoneTemp[BasementZoneTemp].plot(figsize=(25,10))

ZoneTemp[GroundFloorZoneTemp].truncate(before='2013-03-10',after='2013-03-14').plot(figsize=(25,10))

ZoneTemp[Floor1ZoneTemp].plot(figsize=(25,10))

ZoneTemp[Floor2ZoneTemp].plot(figsize=(25,10))

SimulationData['Environment:Outdoor Dry Bulb [C](Hourly)'].truncate(before='2013-03-10',after='2013-03-14').plot(figsize=(25,10))

Floor2Temps = ZoneTemp[Floor2ZoneTemp]

Floor2Temps.info()

Floor2Temps.describe()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import os
import matplotlib.dates as mdates
import datetime as dt

Floor1Energy = list(ColumnsList[(ColumnsList.str.endswith("Total Heating Energy {J}(Hourly)"))&(ColumnsList.str.contains("01"))])

Floor1Energy

df_hourly = SimulationData.resample('H').mean()

#Add the Date and time for pivoting
df_hourly['Date'] = df_hourly.index.map(lambda t: t.date())
df_hourly['Time'] = df_hourly.index.map(lambda t: t.time())

numberofplots = len(Floor1Energy)

pointcounter = 1
fig = plt.figure(figsize=(40, 4 * numberofplots))


for energypoint in Floor1Energy:
    
    print "Loading data from "+energypoint
    
    #Pivot
    df_pivot = pd.pivot_table(df_hourly, values=energypoint, index='Time', columns='Date')

    # Get the data
    x = mdates.drange(df_pivot.columns[0], df_pivot.columns[-1] + datetime.timedelta(days=1), dt.timedelta(days=1))
    y = np.linspace(1, 24, 24)

    # Plot
    ax = fig.add_subplot(numberofplots, 1, pointcounter)

    data = np.ma.masked_invalid(np.array(df_pivot))

    qmesh = ax.pcolormesh(x, y, data)
    cbar = fig.colorbar(qmesh, ax=ax)
    cbar.ax.tick_params(labelsize= 24)
    ax.axis('tight')

    try:
        plt.title(energypoint, fontsize=26)

    except IndexError:
        continue

    # Set up as dates
    ax.xaxis_date()
    fig.autofmt_xdate()
    fig.subplots_adjust(hspace=.5)
    
    pointcounter += 1



